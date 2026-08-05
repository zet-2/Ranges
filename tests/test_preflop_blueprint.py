"""Tests for preflop blueprint loading and traversal."""

import argparse
import copy
from dataclasses import FrozenInstanceError
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

from preflop_blueprint import (
    BlueprintAction,
    BlueprintCacheError,
    BlueprintManifest,
    BlueprintNetworkDisabledError,
    BlueprintNode,
    BlueprintSpot,
    BlueprintValidationError,
    CANONICAL_HAND_CLASSES,
    PokerStudyBlueprintStore,
    build_cli_parser,
    canonical_hand_classes,
    hand_class_combo_count,
    main,
    validate_hand_class,
)


GENERATED_AT = "2026-06-09T19:18:20.659Z"


def manifest_payload():
    return {
        "game": "nl",
        "version": 2,
        "generatedAt": GENERATED_AT,
        "bundleCount": 1,
        "stacks": [100],
        "positions": ["UTG", "BB"],
    }


def spots_payload():
    return {
        "game": "nl",
        "version": 2,
        "stack": 100,
        "generatedAt": GENERATED_AT,
        "spots": [{"history": "UTG", "depth": 1}],
    }


def node_payload():
    return {
        "game": "nl",
        "version": 2,
        "stack": 100,
        "history": "UTG",
        "actor": "UTG",
        "actions": [
            {
                "action": "Fold",
                "kind": "fold",
                "combos": 12,
                "weights": {"72o": 1},
                "evs": {"72o": 0},
            },
            {
                "action": "Call",
                "kind": "call",
                "combos": 1,
                "weights": {"AKs": Decimal("0.25")},
                "evs": {"AKs": Decimal("1.2")},
            },
            {
                "action": "60%",
                "kind": "raise",
                "sizePct": 60,
                "combos": 8,
                "weights": {"AA": 1, "AKs": Decimal("0.5")},
                "evs": {"AA": Decimal("3.4"), "AKs": Decimal("1.3")},
            },
        ],
        "continuingCombos": 9,
    }


class FakeFetcher:
    def __init__(self, *, manifest=None, spots=None, nodes=None):
        self.manifest_payload = manifest or manifest_payload()
        self.spots_payload = spots or spots_payload()
        self.nodes = nodes or {"UTG": node_payload()}
        self.calls = []

    def __call__(self, url, timeout_seconds):
        self.calls.append((url, timeout_seconds))
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.path.endswith("/spots"):
            return copy.deepcopy(self.spots_payload)
        if parsed.path.endswith("/node"):
            return copy.deepcopy(self.nodes[query["history"][0]])
        return copy.deepcopy(self.manifest_payload)


class CanonicalGridTests(unittest.TestCase):
    def test_generates_exact_canonical_169_grid(self):
        self.assertEqual(len(CANONICAL_HAND_CLASSES), 169)
        self.assertEqual(len(set(CANONICAL_HAND_CLASSES)), 169)
        self.assertIs(canonical_hand_classes(), CANONICAL_HAND_CLASSES)
        for expected in ("AA", "AKs", "AKo", "32s", "32o", "22"):
            self.assertIn(expected, CANONICAL_HAND_CLASSES)

    def test_validates_classes_and_combo_multiplicity(self):
        self.assertEqual(validate_hand_class("AKs"), "AKs")
        self.assertEqual(hand_class_combo_count("AA"), 6)
        self.assertEqual(hand_class_combo_count("AKs"), 4)
        self.assertEqual(hand_class_combo_count("AKo"), 12)
        for invalid in ("KAo", "AK", "A1s", "aks", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(BlueprintValidationError):
                    validate_hand_class(invalid)


class ImmutableModelTests(unittest.TestCase):
    def test_models_are_frozen_and_normalized(self):
        manifest = BlueprintManifest(
            "nl", 2, GENERATED_AT, (100,), ("UTG", "BB")
        )
        spot = BlueprintSpot("UTG", 1)
        action = BlueprintAction(
            "60%",
            "RAISE",
            Decimal("60.0"),
            Decimal("8"),
            (("AKs", Decimal("0.5")), ("AA", Decimal(1))),
            (("AKs", Decimal("1.3")), ("AA", Decimal("3.4"))),
        )
        self.assertEqual(action.kind, "raise")
        self.assertEqual(action.weights[0][0], "AA")
        self.assertEqual(manifest.stacks, (100,))
        self.assertEqual(spot.depth, 1)
        with self.assertRaises(FrozenInstanceError):
            action.kind = "fold"

    def test_direct_node_rejects_per_hand_sum_above_one(self):
        fold = BlueprintAction(
            "Fold", "fold", None, Decimal(6), (("AA", Decimal(1)),)
        )
        call = BlueprintAction(
            "Call", "call", None, Decimal(3), (("AA", Decimal("0.5")),)
        )
        with self.assertRaises(BlueprintValidationError):
            BlueprintNode(
                "nl",
                2,
                100,
                "UTG",
                "UTG",
                (fold, call),
                Decimal(3),
                "0" * 64,
            )

    def test_direct_node_accepts_only_small_published_rounding_drift(self):
        fold = BlueprintAction(
            "Fold", "fold", None, Decimal(3), (("AA", Decimal("0.505")),)
        )
        call = BlueprintAction(
            "Call", "call", None, Decimal(3), (("AA", Decimal("0.5")),)
        )
        node = BlueprintNode(
            "nl",
            2,
            100,
            "UTG",
            "UTG",
            (fold, call),
            Decimal(3),
            "0" * 64,
        )
        self.assertEqual(sum(dict(action.weights)["AA"] for action in node.actions), Decimal("1.005"))

        excessive = BlueprintAction(
            "60%", "raise", Decimal(60), Decimal("0.12"), (("AA", Decimal("0.02")),)
        )
        with self.assertRaises(BlueprintValidationError):
            BlueprintNode(
                "nl",
                2,
                100,
                "UTG",
                "UTG",
                (fold, call, excessive),
                Decimal("3.12"),
                "0" * 64,
            )

    def test_direct_node_accepts_only_small_combo_rounding_drift(self):
        fold = BlueprintAction(
            "Fold", "fold", None, Decimal(6), (("72o", Decimal("0.5")),)
        )
        call = BlueprintAction(
            "Call", "call", None, Decimal(3), (("AA", Decimal("0.5")),)
        )
        accepted = BlueprintNode(
            "nl", 2, 100, "UTG", "UTG", (fold, call), Decimal("3.06"), "0" * 64
        )
        self.assertEqual(accepted.continuing_combos, Decimal("3.06"))
        with self.assertRaises(BlueprintValidationError):
            BlueprintNode(
                "nl",
                2,
                100,
                "UTG",
                "UTG",
                (fold, call),
                Decimal("3.11"),
                "0" * 64,
            )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_store(self, fetcher):
        return PokerStudyBlueprintStore(
            self.cache_dir,
            allow_network=True,
            timeout_seconds=Decimal("1.5"),
            fetch_json=fetcher,
        )

    def test_node_roundtrip_and_cache_hit_are_exact(self):
        fetcher = FakeFetcher()
        fresh = self.make_store(fetcher)
        manifest = fresh.manifest()
        spots = fresh.spots(100)
        node = fresh.node(100, "UTG")

        self.assertEqual(manifest.version, 2)
        self.assertEqual(spots, (BlueprintSpot("UTG", 1),))
        self.assertEqual(node.actor, "UTG")
        self.assertEqual(node.continuing_combos, Decimal(9))
        self.assertEqual(len(node.response_sha256), 64)
        self.assertEqual(len(fetcher.calls), 3)

        cached = PokerStudyBlueprintStore(self.cache_dir, allow_network=False)
        self.assertEqual(cached.manifest(), manifest)
        self.assertEqual(cached.spots(100), spots)
        self.assertEqual(cached.node(100, "UTG"), node)
        self.assertEqual(
            cached.validate_cache(100),
            {"manifests": 1, "spot_responses": 1, "nodes": 1},
        )
        self.assertFalse(list(self.cache_dir.rglob("*.tmp")))

    def test_one_argument_fetcher_is_supported(self):
        fake = FakeFetcher()

        def fetch(url):
            return fake(url, 2)

        store = PokerStudyBlueprintStore(
            self.cache_dir, allow_network=True, fetch_json=fetch
        )
        self.assertEqual(store.node(100, "UTG").history, "UTG")

    def test_network_disabled_is_fail_closed_on_miss(self):
        store = PokerStudyBlueprintStore(self.cache_dir)
        with self.assertRaises(BlueprintNetworkDisabledError):
            store.manifest()

    def test_stack_and_history_are_exact_without_interpolation(self):
        store = self.make_store(FakeFetcher())
        store.manifest()
        with self.assertRaisesRegex(BlueprintValidationError, "interpolation"):
            store.spots(70)
        with self.assertRaisesRegex(BlueprintValidationError, "exact documented spot"):
            store.node(100, "BB")

    def test_corrupt_cache_is_not_refetched_or_trusted(self):
        fetcher = FakeFetcher()
        store = self.make_store(fetcher)
        store.node(100, "UTG")
        node_path = store.node_cache_path(100, "UTG")
        envelope = json.loads(node_path.read_text(encoding="utf-8"))
        envelope["payload"]["actor"] = "BB"
        node_path.write_text(json.dumps(envelope), encoding="utf-8")

        reader = PokerStudyBlueprintStore(self.cache_dir, allow_network=False)
        with self.assertRaisesRegex(BlueprintCacheError, "checksum mismatch"):
            reader.node(100, "UTG")

    def test_manifest_mismatch_in_spots_is_rejected_before_cache_write(self):
        payload = spots_payload()
        payload["version"] = 3
        store = self.make_store(FakeFetcher(spots=payload))
        with self.assertRaisesRegex(BlueprintValidationError, "game/version"):
            store.spots(100)
        self.assertFalse(store.spots_cache_path(100).exists())

    def test_node_identity_mismatch_is_rejected(self):
        payload = node_payload()
        payload["history"] = "BB"
        payload["actor"] = "BB"
        store = self.make_store(FakeFetcher(nodes={"UTG": payload}))
        with self.assertRaisesRegex(BlueprintValidationError, "history mismatches"):
            store.node(100, "UTG")

    def test_invalid_hand_class_probability_and_combo_total_are_rejected(self):
        mutations = []

        invalid_class = node_payload()
        invalid_class["actions"][1]["weights"] = {"A1s": Decimal("0.25")}
        invalid_class["actions"][1]["evs"] = {"A1s": 1}
        mutations.append(invalid_class)

        invalid_probability = node_payload()
        invalid_probability["actions"][1]["weights"]["AKs"] = Decimal("1.01")
        invalid_probability["actions"][1]["combos"] = Decimal("4.04")
        mutations.append(invalid_probability)

        invalid_combos = node_payload()
        invalid_combos["actions"][1]["combos"] = 4
        mutations.append(invalid_combos)

        for index, payload in enumerate(mutations):
            with self.subTest(index=index):
                child_cache = self.cache_dir / str(index)
                store = PokerStudyBlueprintStore(
                    child_cache,
                    allow_network=True,
                    fetch_json=FakeFetcher(nodes={"UTG": payload}),
                )
                with self.assertRaises(BlueprintValidationError):
                    store.node(100, "UTG")
                self.assertFalse(store.node_cache_path(100, "UTG").exists())

    def test_per_hand_action_sum_and_unknown_kind_are_rejected(self):
        excessive = node_payload()
        excessive["actions"][1]["weights"] = {"AA": Decimal("0.5")}
        excessive["actions"][1]["evs"] = {"AA": 1}
        excessive["actions"][1]["combos"] = 3
        excessive["continuingCombos"] = 11

        unknown = node_payload()
        unknown["actions"][1]["kind"] = "check"

        for name, payload in (("sum", excessive), ("kind", unknown)):
            with self.subTest(name=name):
                store = PokerStudyBlueprintStore(
                    self.cache_dir / name,
                    allow_network=True,
                    fetch_json=FakeFetcher(nodes={"UTG": payload}),
                )
                with self.assertRaises(BlueprintValidationError):
                    store.node(100, "UTG")


class CliTests(unittest.TestCase):
    def test_parser_and_minimal_offline_sync_then_validate(self):
        parser = build_cli_parser()
        args = parser.parse_args(
            [
                "sync",
                "--cache-dir",
                "somewhere",
                "--stack",
                "100",
                "--max-depth",
                "1",
                "--workers",
                "1",
            ]
        )
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual((args.stack, args.max_depth, args.workers), (100, 1, 1))

        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            result = main(
                [
                    "sync",
                    "--cache-dir",
                    temporary,
                    "--stack",
                    "100",
                    "--max-depth",
                    "1",
                    "--workers",
                    "1",
                ],
                fetch_json=FakeFetcher(),
                stdout=output,
            )
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.getvalue())["synced_nodes"], 1)

            validation_output = io.StringIO()
            result = main(
                ["validate", "--cache-dir", temporary, "--stack", "100"],
                stdout=validation_output,
            )
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(validation_output.getvalue())["nodes"], 1)


if __name__ == "__main__":
    unittest.main()
