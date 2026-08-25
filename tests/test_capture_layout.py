#!/usr/bin/env python3
"""Regression tests for versioned native-HU capture calibration."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import poker_assistant as app
import calibrate
from capture_layout import (
    CaptureLayoutError,
    load_capture_layout,
    parse_capture_layout,
    sha256_file,
    verified_payload,
)


def draft_payload() -> dict:
    regions = {
        "seat1": {"left": 100, "top": 80, "width": 220, "height": 120},
        "hero": {"left": 400, "top": 500, "width": 260, "height": 150},
        "board": {"left": 330, "top": 260, "width": 340, "height": 160},
        "buttons": {"left": 500, "top": 700, "width": 450, "height": 140},
    }
    return {
        "schema_version": 1,
        "artifact_type": "pokerstars_capture_layout",
        "layout_id": "pokerstars.hu.owned-sim.v1",
        "table_format": "heads_up",
        "monitor": {"index": 1, "width_px": 1200, "height_px": 900},
        "canonical_seats": {"hero": 4, "opponent": 0},
        "regions": regions,
        "calibration": {
            "method": "interactive_region_corners",
            "status": "draft",
            "reference_frame_sha256": "a" * 64,
            "crop_sha256": {
                name: str(index + 1) * 64
                for index, name in enumerate(sorted(regions))
            },
            "reviewed_reference_frame_sha256": None,
        },
    }


class CaptureLayoutContractTests(unittest.TestCase):
    def test_native_hu_profile_maps_only_s0_and_s4(self):
        layout = parse_capture_layout(draft_payload())
        self.assertEqual("heads_up", layout.table_format)
        self.assertEqual(frozenset({0, 4}), layout.active_seat_indices)
        self.assertEqual({"seat1": 0}, layout.opponent_capture_seats)
        self.assertEqual(set(draft_payload()["regions"]), set(layout.region_map))
        self.assertEqual(64, len(layout.fingerprint))
        layout.assert_monitor(1, 1200, 900)
        with self.assertRaisesRegex(CaptureLayoutError, "monitor changed"):
            layout.assert_monitor(2, 1200, 900)

    def test_draft_is_not_live_eligible_until_exact_frame_is_reviewed(self):
        layout = parse_capture_layout(draft_payload())
        with self.assertRaisesRegex(CaptureLayoutError, "is draft"):
            layout.require_verified()

        approved_payload = verified_payload(layout)
        approved = parse_capture_layout(approved_payload)
        approved.require_verified()
        self.assertEqual(
            approved.calibration.reference_frame_sha256,
            approved.calibration.reviewed_reference_frame_sha256,
        )

    def test_rejects_unknown_fields_wrong_slots_bounds_and_hash_coverage(self):
        unknown = draft_payload()
        unknown["surprise"] = True
        with self.assertRaisesRegex(CaptureLayoutError, "keys mismatch"):
            parse_capture_layout(unknown)

        wrong_hero = draft_payload()
        wrong_hero["canonical_seats"]["hero"] = 1
        with self.assertRaisesRegex(CaptureLayoutError, "Hero must remain S4"):
            parse_capture_layout(wrong_hero)

        out_of_bounds = draft_payload()
        out_of_bounds["regions"]["buttons"]["height"] = 300
        with self.assertRaisesRegex(CaptureLayoutError, "monitor height"):
            parse_capture_layout(out_of_bounds)

        missing_hash = draft_payload()
        del missing_hash["calibration"]["crop_sha256"]["hero"]
        with self.assertRaisesRegex(CaptureLayoutError, "exactly cover"):
            parse_capture_layout(missing_hash)

    def test_loader_rejects_duplicate_keys_and_draft_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "draft.json"
            profile.write_text(json.dumps(draft_payload()), encoding="utf-8")
            with self.assertRaisesRegex(CaptureLayoutError, "is draft"):
                load_capture_layout(profile)
            self.assertEqual(
                "draft",
                load_capture_layout(profile, require_verified=False).calibration.status,
            )

            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(CaptureLayoutError, "duplicate JSON key"):
                load_capture_layout(duplicate, require_verified=False)


class NativeHUCanonicalVisionTests(unittest.TestCase):
    @staticmethod
    def captures() -> dict[str, Image.Image]:
        return {
            "seat1": Image.new("RGB", (220, 120), "red"),
            "hero": Image.new("RGB", (260, 150), "green"),
            "board": Image.new("RGB", (340, 160), "blue"),
            "buttons": Image.new("RGB", (450, 140), "yellow"),
        }

    @staticmethod
    def wire_payload() -> dict:
        return {
            "n": ["Villain", "", "", "", "Hero", ""],
            "s": [100, 0, 0, 0, 100, 0],
            "w": [1, 0, 0, 0, 0.5, 0],
            "x": ["A", "E", "E", "E", "A", "E"],
            "d": 4,
            "v": ["", "", "", "", "", ""],
            "h": ["As", "Kd"],
            "b": [],
            "p": 1.5,
            "o": ["FOLD", "CALL", "RAISE"],
            "c": 0.5,
        }

    def test_hu_mosaic_uses_explicit_blanks_and_layout_instruction(self):
        with mock.patch.multiple(
            app,
            CAPTURE_TABLE_FORMAT="heads_up",
            CAPTURE_ACTIVE_SEAT_INDICES=frozenset({0, 4}),
        ):
            mosaic = app.build_vision_mosaic(self.captures())
            self.assertEqual((1080, 600), mosaic.size)
            contents, _ = app.build_vision_request(self.captures())
            self.assertIn("only S0 (opponent) and S4", contents[0])
            self.assertIn("only S0 and S4", app.build_parallel_strategy_prompt())

    def test_blank_detection_uses_the_shared_canonical_seat_map(self):
        with mock.patch.multiple(
            app,
            CAPTURE_TABLE_FORMAT="heads_up",
            CAPTURE_ACTIVE_SEAT_INDICES=frozenset({0, 4}),
            CANONICAL_SEAT_CAPTURES={"seat2": 4},
        ):
            with self.assertRaisesRegex(KeyError, "missing calibrated capture"):
                app.capture_or_explicit_blank({}, "seat2")

    def test_screen_grab_delegates_monitor_validation_to_the_profile(self):
        profile = mock.Mock(unsafe=True)
        profile.assert_monitor.side_effect = CaptureLayoutError("monitor changed")
        screen = mock.MagicMock()
        screen.__enter__.return_value.monitors = [
            {},
            {"left": 0, "top": 0, "width": 1200, "height": 900},
        ]
        with (
            mock.patch.object(app, "CAPTURE_LAYOUT_PROFILE", profile),
            mock.patch.object(app.mss, "MSS", return_value=screen),
        ):
            with self.assertRaisesRegex(CaptureLayoutError, "monitor changed"):
                app.grab_zone_set(
                    1,
                    {"hero": {"left": 1, "top": 1, "width": 10, "height": 10}},
                )
        profile.assert_monitor.assert_called_once_with(1, 1200, 900)

    def test_hu_wire_contract_rejects_content_in_inactive_slots(self):
        with mock.patch.multiple(
            app,
            CAPTURE_TABLE_FORMAT="heads_up",
            CAPTURE_ACTIVE_SEAT_INDICES=frozenset({0, 4}),
            CAPTURE_LAYOUT_FINGERPRINT="a" * 64,
        ):
            expanded = app.expand_fast_vision_payload(self.wire_payload())
            live = [
                seat["seat_index"]
                for seat in expanded["seats"]
                if not seat["is_empty"]
            ]
            self.assertEqual([0, 4], live)
            parsed = app.parse_response(
                json.dumps(self.wire_payload()),
                locally_dealt_seats={0},
                hero_turn_confirmed=True,
            )
            self.assertEqual(2, parsed.meta_info.table_size)
            self.assertEqual("heads_up", parsed.capture_table_format)
            self.assertEqual("a" * 64, parsed.capture_layout_fingerprint)

            hallucinated = copy.deepcopy(self.wire_payload())
            hallucinated["n"][2] = "Ghost"
            with self.assertRaisesRegex(ValueError, "inactive native-HU slot S2"):
                app.expand_fast_vision_payload(hallucinated)

            bad_dealer = copy.deepcopy(self.wire_payload())
            bad_dealer["d"] = 1
            with self.assertRaisesRegex(ValueError, "dealer must be"):
                app.expand_fast_vision_payload(bad_dealer)


class CalibrationApprovalTests(unittest.TestCase):
    def _write_real_draft(self, directory: Path) -> tuple[Path, Path]:
        evidence = directory / "evidence"
        evidence.mkdir()
        reference = Image.new("RGB", (1200, 900), (12, 34, 56))
        # Give each semantic area different pixels so swapped crops are visible.
        pixels = reference.load()
        payload = draft_payload()
        colors = {
            "seat1": (200, 20, 20),
            "hero": (20, 200, 20),
            "board": (20, 20, 200),
            "buttons": (200, 200, 20),
        }
        for name, region in payload["regions"].items():
            for y in range(region["top"], region["top"] + region["height"]):
                for x in range(region["left"], region["left"] + region["width"]):
                    pixels[x, y] = colors[name]
        reference_path = evidence / "reference.png"
        reference.save(reference_path)
        crop_hashes = {}
        for name, region in payload["regions"].items():
            crop = reference.crop(
                (
                    region["left"],
                    region["top"],
                    region["left"] + region["width"],
                    region["top"] + region["height"],
                )
            )
            crop_path = evidence / f"{name}.png"
            crop.save(crop_path)
            crop_hashes[name] = sha256_file(crop_path)
        payload["calibration"]["reference_frame_sha256"] = sha256_file(
            reference_path
        )
        payload["calibration"]["crop_sha256"] = crop_hashes
        draft = directory / "draft.json"
        draft.write_text(json.dumps(payload), encoding="utf-8")
        return draft, evidence

    def test_approve_cli_rechecks_pixels_and_writes_verified_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            draft, evidence = self._write_real_draft(directory)
            output = directory / "verified.json"
            status = calibrate.main(
                [
                    "approve",
                    "--profile",
                    str(draft),
                    "--evidence-dir",
                    str(evidence),
                    "--output",
                    str(output),
                    "--i-reviewed-crops",
                ]
            )
            self.assertEqual(0, status)
            load_capture_layout(output).require_verified()

    def test_approve_cli_refuses_a_tampered_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            draft, evidence = self._write_real_draft(directory)
            Image.new("RGB", (220, 120), "black").save(evidence / "seat1.png")
            status = calibrate.main(
                [
                    "approve",
                    "--profile",
                    str(draft),
                    "--evidence-dir",
                    str(evidence),
                    "--output",
                    str(directory / "verified.json"),
                    "--i-reviewed-crops",
                ]
            )
            self.assertEqual(2, status)


if __name__ == "__main__":
    unittest.main()
