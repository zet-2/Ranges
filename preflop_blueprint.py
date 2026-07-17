"""Validated, local cache for PokerStudy's public NLHE preflop blueprint.

The module deliberately stores only responses fetched from the documented
``/api/ranges/nl/v2`` API.  It neither embeds strategy data nor guesses a
nearby stack or action path.  Cached responses are canonicalized, checksummed,
and revalidated every time they are read from disk.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable, Iterable, Mapping, Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://www.pokerstudy.ai/api/ranges/nl/v2"
EXPECTED_GAME = "nl"
EXPECTED_VERSION = 2
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
# ``combos`` and ``continuingCombos`` are separately rounded in the source.
# Across the documented 100 BB depth<=2 corpus the largest observed delta is
# 0.06 combos, so 0.10 accepts source rounding while still rejecting a material
# range-total mismatch.
COMBO_TOLERANCE = Decimal("0.10")
# The public artifacts round individual action weights to 0.005 increments.
# With several actions at one node the published values can therefore sum to
# 1.005 for a hand class.  Accept only that documented-size rounding drift;
# materially inconsistent nodes still fail closed.
FREQUENCY_TOLERANCE = Decimal("0.01")

_RANKS = "AKQJT98765432"
_POSITION_RE = re.compile(r"^[A-Z][A-Z0-9]*$")
_HISTORY_RE = re.compile(r"^[A-Za-z0-9%.:-]+(?:_[A-Za-z0-9%.:-]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_ACTION_KINDS = frozenset({"fold", "call", "raise", "allin"})


class BlueprintError(RuntimeError):
    """Base error for blueprint retrieval and validation."""


class BlueprintValidationError(BlueprintError, ValueError):
    """The remote or cached payload violates the documented API contract."""


class BlueprintCacheError(BlueprintValidationError):
    """A cache entry is truncated, corrupt, or stored under the wrong key."""


class BlueprintNetworkDisabledError(BlueprintError):
    """A requested object is not cached and network access is disabled."""


class BlueprintFetchError(BlueprintError):
    """The public API could not be fetched or decoded."""


def _generate_hand_classes() -> tuple[str, ...]:
    classes: list[str] = []
    for row, first in enumerate(_RANKS):
        for column, second in enumerate(_RANKS):
            if row == column:
                classes.append(first + second)
            elif row < column:
                classes.append(first + second + "s")
            else:
                classes.append(second + first + "o")
    if len(classes) != 169 or len(set(classes)) != 169:  # pragma: no cover
        raise AssertionError("canonical Hold'em grid generation failed")
    return tuple(classes)


CANONICAL_HAND_CLASSES = _generate_hand_classes()
_HAND_CLASS_SET = frozenset(CANONICAL_HAND_CLASSES)
_HAND_CLASS_ORDER = {
    hand_class: index for index, hand_class in enumerate(CANONICAL_HAND_CLASSES)
}


def canonical_hand_classes() -> tuple[str, ...]:
    """Return the canonical 13x13 Hold'em starting-hand grid."""

    return CANONICAL_HAND_CLASSES


def validate_hand_class(value: object) -> str:
    """Return a canonical 169-grid class or raise fail-closed."""

    if not isinstance(value, str) or value not in _HAND_CLASS_SET:
        raise BlueprintValidationError(
            f"invalid Hold'em hand class {value!r}; expected one of the 169 canonical classes"
        )
    return value


def hand_class_combo_count(hand_class: str) -> int:
    """Return concrete combo multiplicity: pair=6, suited=4, offsuit=12."""

    hand_class = validate_hand_class(hand_class)
    if len(hand_class) == 2:
        return 6
    return 4 if hand_class.endswith("s") else 12


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BlueprintValidationError(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BlueprintValidationError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _as_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise BlueprintValidationError(f"{field} must be numeric, not boolean")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BlueprintValidationError(f"{field} must be finite")
        value = str(value)
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BlueprintValidationError(f"{field} must be a decimal number") from error
    if not result.is_finite():
        raise BlueprintValidationError(f"{field} must be finite")
    return Decimal(0) if result == 0 else result.normalize()


def _validate_history(value: object, field: str = "history") -> str:
    history = _require_nonempty_string(value, field)
    if len(history) > 8192 or not _HISTORY_RE.fullmatch(history):
        raise BlueprintValidationError(f"{field} is not a canonical action path")
    return history


def _decimal_json(value: Decimal) -> str:
    if not value.is_finite():
        raise BlueprintValidationError("canonical JSON cannot contain non-finite Decimal")
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text


def _canonical_json(value: object) -> str:
    """Serialize JSON deterministically while preserving Decimal as a number."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return _decimal_json(value)
    if isinstance(value, float):
        return _decimal_json(_as_decimal(value, "JSON number"))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise BlueprintValidationError("canonical JSON object keys must be strings")
        return "{" + ",".join(
            _canonical_json(key) + ":" + _canonical_json(value[key])
            for key in sorted(value)
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    raise BlueprintValidationError(
        f"unsupported canonical JSON value {type(value).__name__}"
    )


def canonical_response_sha256(payload: object) -> str:
    """Return SHA-256 of the canonical response representation."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BlueprintValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise BlueprintValidationError(f"non-finite JSON number {value!r}")


def _strict_json_loads(data: str | bytes, source: str) -> object:
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(
            data,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except BlueprintValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BlueprintValidationError(f"{source} is not valid UTF-8 JSON") from error


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value.keys()
    ):
        raise BlueprintValidationError(f"{field} must be a JSON object")
    return value  # type: ignore[return-value]


def _ordered_numeric_map(
    value: object,
    field: str,
    *,
    probability: bool,
) -> tuple[tuple[str, Decimal], ...]:
    source = _require_mapping(value, field)
    parsed: list[tuple[str, Decimal]] = []
    for hand_class, raw_value in source.items():
        canonical = validate_hand_class(hand_class)
        decimal_value = _as_decimal(raw_value, f"{field}.{canonical}")
        if probability and not Decimal(0) <= decimal_value <= Decimal(1):
            raise BlueprintValidationError(
                f"{field}.{canonical} must be between 0 and 1"
            )
        parsed.append((canonical, decimal_value))
    parsed.sort(key=lambda pair: _HAND_CLASS_ORDER[pair[0]])
    return tuple(parsed)


def _normalize_numeric_pairs(
    pairs: Iterable[tuple[str, object]],
    field: str,
    *,
    probability: bool,
) -> tuple[tuple[str, Decimal], ...]:
    parsed: list[tuple[str, Decimal]] = []
    seen: set[str] = set()
    try:
        sequence = tuple(pairs)
    except TypeError as error:
        raise BlueprintValidationError(f"{field} must contain hand/value pairs") from error
    for item in sequence:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise BlueprintValidationError(f"{field} must contain hand/value pairs")
        hand_class = validate_hand_class(item[0])
        if hand_class in seen:
            raise BlueprintValidationError(f"{field} repeats {hand_class}")
        seen.add(hand_class)
        number = _as_decimal(item[1], f"{field}.{hand_class}")
        if probability and not Decimal(0) <= number <= Decimal(1):
            raise BlueprintValidationError(
                f"{field}.{hand_class} must be between 0 and 1"
            )
        parsed.append((hand_class, number))
    parsed.sort(key=lambda pair: _HAND_CLASS_ORDER[pair[0]])
    return tuple(parsed)


def _weighted_combo_total(weights: Iterable[tuple[str, Decimal]]) -> Decimal:
    return sum(
        (Decimal(hand_class_combo_count(hand_class)) * weight for hand_class, weight in weights),
        Decimal(0),
    )


@dataclass(frozen=True)
class BlueprintManifest:
    game: str
    version: int
    generated_at: str
    stacks: tuple[int, ...]
    positions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.game != EXPECTED_GAME:
            raise BlueprintValidationError(f"manifest game must be {EXPECTED_GAME!r}")
        if self.version != EXPECTED_VERSION:
            raise BlueprintValidationError(
                f"manifest version must be {EXPECTED_VERSION}"
            )
        generated_at = _require_nonempty_string(self.generated_at, "generated_at")
        try:
            parsed_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise BlueprintValidationError(
                "generated_at must be an ISO-8601 timestamp"
            ) from error
        if parsed_time.tzinfo is None:
            raise BlueprintValidationError("generated_at must include a timezone")

        stacks = tuple(self.stacks)
        if not stacks or any(
            isinstance(stack, bool) or not isinstance(stack, int) or stack <= 0
            for stack in stacks
        ):
            raise BlueprintValidationError("manifest stacks must be positive integers")
        if stacks != tuple(sorted(set(stacks))):
            raise BlueprintValidationError(
                "manifest stacks must be unique and strictly increasing"
            )
        positions = tuple(self.positions)
        if not positions or any(
            not isinstance(position, str) or not _POSITION_RE.fullmatch(position)
            for position in positions
        ):
            raise BlueprintValidationError("manifest positions are malformed")
        if len(positions) != len(set(positions)):
            raise BlueprintValidationError("manifest positions must be unique")
        object.__setattr__(self, "stacks", stacks)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class BlueprintSpot:
    history: str
    depth: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", _validate_history(self.history))
        object.__setattr__(self, "depth", _require_int(self.depth, "depth", minimum=1))


@dataclass(frozen=True)
class BlueprintAction:
    label: str
    kind: str
    size_pct: Decimal | None
    combos: Decimal
    weights: tuple[tuple[str, Decimal], ...]
    evs: tuple[tuple[str, Decimal], ...] | None = None

    def __post_init__(self) -> None:
        label = _require_nonempty_string(self.label, "action label")
        kind = _require_nonempty_string(self.kind, "action kind").lower()
        if kind not in _KNOWN_ACTION_KINDS:
            raise BlueprintValidationError(f"unknown action kind {kind!r}")
        size_pct = (
            None
            if self.size_pct is None
            else _as_decimal(self.size_pct, "action size_pct")
        )
        if kind == "raise":
            if size_pct is None or size_pct <= 0:
                raise BlueprintValidationError("raise action requires positive size_pct")
            expected_label = _decimal_json(size_pct) + "%"
            if label != expected_label:
                raise BlueprintValidationError(
                    f"raise label {label!r} does not match sizePct {expected_label!r}"
                )
        elif size_pct is not None:
            raise BlueprintValidationError(f"{kind} action must not include size_pct")
        if kind == "fold" and label != "Fold":
            raise BlueprintValidationError("fold action label must be 'Fold'")
        if kind == "call" and label != "Call":
            raise BlueprintValidationError("call action label must be 'Call'")
        if kind == "allin" and label != "AI":
            raise BlueprintValidationError("allin action label must be 'AI'")

        combos = _as_decimal(self.combos, "action combos")
        if not Decimal(0) <= combos <= Decimal(1326):
            raise BlueprintValidationError("action combos must be between 0 and 1326")
        weights = _normalize_numeric_pairs(
            self.weights, "action weights", probability=True
        )
        expected_combos = _weighted_combo_total(weights)
        if abs(expected_combos - combos) > COMBO_TOLERANCE:
            raise BlueprintValidationError(
                f"action combos {combos} disagree with weighted total {expected_combos}"
            )

        evs: tuple[tuple[str, Decimal], ...] | None
        if self.evs is None:
            evs = None
        else:
            evs = _normalize_numeric_pairs(self.evs, "action evs", probability=False)
            if {hand for hand, _ in evs} != {hand for hand, _ in weights}:
                raise BlueprintValidationError(
                    "action evs must cover exactly the hand classes in weights"
                )

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_pct", size_pct)
        object.__setattr__(self, "combos", combos)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "evs", evs)


@dataclass(frozen=True)
class BlueprintNode:
    game: str
    version: int
    stack: int
    history: str
    actor: str
    actions: tuple[BlueprintAction, ...]
    continuing_combos: Decimal
    response_sha256: str

    def __post_init__(self) -> None:
        if self.game != EXPECTED_GAME or self.version != EXPECTED_VERSION:
            raise BlueprintValidationError("node game/version does not match NL v2")
        stack = _require_int(self.stack, "node stack", minimum=1)
        history = _validate_history(self.history)
        actor = _require_nonempty_string(self.actor, "node actor")
        if not _POSITION_RE.fullmatch(actor) or history.split("_")[-1] != actor:
            raise BlueprintValidationError(
                "node actor must be the final position in history"
            )
        actions = tuple(self.actions)
        if not actions or any(not isinstance(action, BlueprintAction) for action in actions):
            raise BlueprintValidationError("node actions must contain BlueprintAction values")
        labels = [action.label for action in actions]
        if len(labels) != len(set(labels)):
            raise BlueprintValidationError("node action labels must be unique")

        per_hand: dict[str, Decimal] = {}
        for action in actions:
            for hand_class, frequency in action.weights:
                per_hand[hand_class] = per_hand.get(hand_class, Decimal(0)) + frequency
        for hand_class, total in per_hand.items():
            if total > Decimal(1) + FREQUENCY_TOLERANCE:
                raise BlueprintValidationError(
                    f"action frequencies for {hand_class} sum to {total}, above 1"
                )

        continuing = _as_decimal(self.continuing_combos, "continuing_combos")
        if not Decimal(0) <= continuing <= Decimal(1326):
            raise BlueprintValidationError(
                "continuing_combos must be between 0 and 1326"
            )
        expected_continuing = sum(
            (action.combos for action in actions if action.kind != "fold"),
            Decimal(0),
        )
        if abs(expected_continuing - continuing) > COMBO_TOLERANCE:
            raise BlueprintValidationError(
                "continuing_combos disagrees with non-fold action ranges"
            )
        digest = _require_nonempty_string(self.response_sha256, "response_sha256")
        if not _SHA256_RE.fullmatch(digest):
            raise BlueprintValidationError(
                "response_sha256 must be a lowercase SHA-256 digest"
            )

        object.__setattr__(self, "stack", stack)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "continuing_combos", continuing)


def _parse_manifest(payload: object) -> BlueprintManifest:
    source = _require_mapping(payload, "manifest response")
    generated_at = source.get("generatedAt")
    raw_stacks = source.get("stacks")
    raw_positions = source.get("positions")
    if not isinstance(raw_stacks, list) or not isinstance(raw_positions, list):
        raise BlueprintValidationError("manifest stacks/positions must be arrays")
    return BlueprintManifest(
        game=source.get("game"),  # type: ignore[arg-type]
        version=source.get("version"),  # type: ignore[arg-type]
        generated_at=generated_at,  # type: ignore[arg-type]
        stacks=tuple(raw_stacks),  # type: ignore[arg-type]
        positions=tuple(raw_positions),  # type: ignore[arg-type]
    )


def _parse_spots(
    payload: object,
    manifest: BlueprintManifest,
    expected_stack: int,
) -> tuple[BlueprintSpot, ...]:
    source = _require_mapping(payload, "spots response")
    if source.get("game") != manifest.game or source.get("version") != manifest.version:
        raise BlueprintValidationError("spots response game/version mismatches manifest")
    if source.get("stack") != expected_stack:
        raise BlueprintValidationError("spots response stack mismatches request")
    if "generatedAt" in source and source["generatedAt"] != manifest.generated_at:
        raise BlueprintValidationError("spots response generatedAt mismatches manifest")
    raw_spots = source.get("spots")
    if not isinstance(raw_spots, list) or not raw_spots:
        raise BlueprintValidationError("spots response must contain a non-empty spots array")
    spots: list[BlueprintSpot] = []
    seen: set[str] = set()
    for index, raw_spot in enumerate(raw_spots):
        item = _require_mapping(raw_spot, f"spots[{index}]")
        spot = BlueprintSpot(
            history=item.get("history"),  # type: ignore[arg-type]
            depth=item.get("depth"),  # type: ignore[arg-type]
        )
        if spot.history in seen:
            raise BlueprintValidationError(f"spots repeats history {spot.history!r}")
        seen.add(spot.history)
        final_position = spot.history.split("_")[-1]
        if final_position not in manifest.positions:
            raise BlueprintValidationError(
                f"spot {spot.history!r} ends in unknown position {final_position!r}"
            )
        spots.append(spot)
    return tuple(spots)


def _parse_action(payload: object, index: int) -> BlueprintAction:
    source = _require_mapping(payload, f"actions[{index}]")
    weights = _ordered_numeric_map(
        source.get("weights"), f"actions[{index}].weights", probability=True
    )
    evs: tuple[tuple[str, Decimal], ...] | None = None
    if "evs" in source:
        evs = _ordered_numeric_map(
            source["evs"], f"actions[{index}].evs", probability=False
        )
    return BlueprintAction(
        label=source.get("action"),  # type: ignore[arg-type]
        kind=source.get("kind"),  # type: ignore[arg-type]
        size_pct=(source.get("sizePct") if "sizePct" in source else None),  # type: ignore[arg-type]
        combos=source.get("combos"),  # type: ignore[arg-type]
        weights=weights,
        evs=evs,
    )


def _parse_node(
    payload: object,
    manifest: BlueprintManifest,
    expected_stack: int,
    expected_history: str,
    response_sha256: str,
) -> BlueprintNode:
    source = _require_mapping(payload, "node response")
    if source.get("game") != manifest.game or source.get("version") != manifest.version:
        raise BlueprintValidationError("node response game/version mismatches manifest")
    if source.get("stack") != expected_stack:
        raise BlueprintValidationError("node response stack mismatches request")
    if source.get("history") != expected_history:
        raise BlueprintValidationError("node response history mismatches request")
    actor = source.get("actor")
    if actor not in manifest.positions:
        raise BlueprintValidationError("node response actor is not in manifest positions")
    raw_actions = source.get("actions")
    if not isinstance(raw_actions, list):
        raise BlueprintValidationError("node response actions must be an array")
    actions = tuple(_parse_action(action, index) for index, action in enumerate(raw_actions))
    return BlueprintNode(
        game=source.get("game"),  # type: ignore[arg-type]
        version=source.get("version"),  # type: ignore[arg-type]
        stack=source.get("stack"),  # type: ignore[arg-type]
        history=source.get("history"),  # type: ignore[arg-type]
        actor=actor,  # type: ignore[arg-type]
        actions=actions,
        continuing_combos=source.get("continuingCombos"),  # type: ignore[arg-type]
        response_sha256=response_sha256,
    )


FetchJson = Callable[..., object]


class PokerStudyBlueprintStore:
    """Cache-first, exact-key client for PokerStudy's NL v2 blueprint.

    ``fetch_json`` may accept ``(url, timeout_seconds)`` or just ``(url)``.
    It is never invoked unless ``allow_network`` is true.  A corrupt existing
    cache entry raises immediately rather than being silently replaced.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        allow_network: bool = False,
        timeout_seconds: float = 2,
        fetch_json: FetchJson | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if not isinstance(allow_network, bool):
            raise BlueprintValidationError("allow_network must be boolean")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float, Decimal)
        ):
            raise BlueprintValidationError("timeout_seconds must be numeric")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise BlueprintValidationError("timeout_seconds must be positive and finite")
        self.allow_network = allow_network
        self.timeout_seconds = timeout
        self._fetch_json = fetch_json or self._default_fetch_json
        self._manifest_memory: BlueprintManifest | None = None
        self._spots_memory: dict[int, tuple[BlueprintSpot, ...]] = {}

    @property
    def manifest_cache_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def spots_cache_path(self, stack: int) -> Path:
        stack = _require_int(stack, "stack", minimum=1)
        return self.cache_dir / "stacks" / str(stack) / "spots.json"

    def node_cache_path(self, stack: int, history: str) -> Path:
        stack = _require_int(stack, "stack", minimum=1)
        history = _validate_history(history)
        history_digest = hashlib.sha256(history.encode("utf-8")).hexdigest()
        return self.cache_dir / "stacks" / str(stack) / "nodes" / f"{history_digest}.json"

    def manifest(self) -> BlueprintManifest:
        if self._manifest_memory is not None:
            return self._manifest_memory
        path = self.manifest_cache_path
        if path.exists():
            payload, _ = self._read_cache(path)
            parsed = _parse_manifest(payload)
        else:
            payload = self._fetch_payload(API_BASE_URL)
            parsed = _parse_manifest(payload)
            self._write_cache(path, payload)
        self._manifest_memory = parsed
        return parsed

    def spots(self, stack: int) -> tuple[BlueprintSpot, ...]:
        stack = self._validate_requested_stack(stack)
        if stack in self._spots_memory:
            return self._spots_memory[stack]
        path = self.spots_cache_path(stack)
        manifest = self.manifest()
        if path.exists():
            payload, _ = self._read_cache(path)
            parsed = _parse_spots(payload, manifest, stack)
        else:
            url = API_BASE_URL + "/spots?" + urlencode({"stack": stack})
            payload = self._fetch_payload(url)
            parsed = _parse_spots(payload, manifest, stack)
            self._write_cache(path, payload)
        self._spots_memory[stack] = parsed
        return parsed

    def node(self, stack: int, history: str) -> BlueprintNode:
        stack = self._validate_requested_stack(stack)
        history = _validate_history(history)
        available = {spot.history for spot in self.spots(stack)}
        if history not in available:
            raise BlueprintValidationError(
                f"history {history!r} is not an exact documented spot for stack {stack}"
            )
        path = self.node_cache_path(stack, history)
        manifest = self.manifest()
        if path.exists():
            payload, digest = self._read_cache(path)
            return _parse_node(payload, manifest, stack, history, digest)
        url = API_BASE_URL + "/node?" + urlencode(
            {"stack": stack, "history": history}
        )
        payload = self._fetch_payload(url)
        digest = canonical_response_sha256(payload)
        parsed = _parse_node(payload, manifest, stack, history, digest)
        self._write_cache(path, payload)
        return parsed

    def validate_cache(self, stack: int | None = None) -> dict[str, int]:
        """Validate every selected cached response without using the network."""

        manifest_payload, _ = self._read_cache(self.manifest_cache_path)
        manifest = _parse_manifest(manifest_payload)
        stacks_root = self.cache_dir / "stacks"
        if stack is not None:
            stack = _require_int(stack, "stack", minimum=1)
            if stack not in manifest.stacks:
                raise BlueprintValidationError(
                    f"stack {stack} is not present in the exact manifest"
                )
            selected_stacks = (stack,)
        elif stacks_root.exists():
            selected: list[int] = []
            for entry in stacks_root.iterdir():
                if not entry.is_dir() or not entry.name.isdigit():
                    raise BlueprintCacheError(
                        f"unexpected cache stack directory {entry.name!r}"
                    )
                cached_stack = int(entry.name)
                if cached_stack not in manifest.stacks:
                    raise BlueprintCacheError(
                        f"cached stack {cached_stack} is absent from manifest"
                    )
                selected.append(cached_stack)
            selected_stacks = tuple(sorted(selected))
        else:
            selected_stacks = ()

        spot_response_count = 0
        node_count = 0
        for cached_stack in selected_stacks:
            spots_path = self.spots_cache_path(cached_stack)
            if not spots_path.is_file():
                raise BlueprintCacheError(
                    f"missing spots cache for stack {cached_stack}"
                )
            spots_payload, _ = self._read_cache(spots_path)
            spots = _parse_spots(spots_payload, manifest, cached_stack)
            spot_response_count += 1
            known_histories = {spot.history for spot in spots}
            nodes_dir = spots_path.parent / "nodes"
            if not nodes_dir.exists():
                continue
            for node_path in sorted(nodes_dir.glob("*.json")):
                payload, digest = self._read_cache(node_path)
                source = _require_mapping(payload, "cached node response")
                history = _validate_history(source.get("history"))
                expected_name = hashlib.sha256(history.encode("utf-8")).hexdigest() + ".json"
                if node_path.name != expected_name:
                    raise BlueprintCacheError(
                        f"node cache filename does not match history {history!r}"
                    )
                if history not in known_histories:
                    raise BlueprintCacheError(
                        f"cached node history {history!r} is absent from spots"
                    )
                _parse_node(payload, manifest, cached_stack, history, digest)
                node_count += 1
        return {
            "manifests": 1,
            "spot_responses": spot_response_count,
            "nodes": node_count,
        }

    def _validate_requested_stack(self, stack: int) -> int:
        stack = _require_int(stack, "stack", minimum=1)
        if stack not in self.manifest().stacks:
            raise BlueprintValidationError(
                f"stack {stack} is not present in the exact manifest; interpolation is disabled"
            )
        return stack

    def _fetch_payload(self, url: str) -> object:
        if not self.allow_network:
            raise BlueprintNetworkDisabledError(
                f"cache miss for {url}; network access is disabled"
            )
        try:
            fetcher = self._fetch_json
            try:
                signature = inspect.signature(fetcher)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                try:
                    signature.bind(url, self.timeout_seconds)
                except TypeError:
                    try:
                        signature.bind(url)
                    except TypeError as error:
                        raise BlueprintFetchError(
                            "fetch_json must accept (url, timeout_seconds) or (url)"
                        ) from error
                    result = fetcher(url)
                else:
                    result = fetcher(url, self.timeout_seconds)
            else:
                result = fetcher(url, self.timeout_seconds)
        except BlueprintError:
            raise
        except Exception as error:
            raise BlueprintFetchError(f"failed to fetch {url}: {error}") from error
        if isinstance(result, (str, bytes)):
            result = _strict_json_loads(result, url)
        # Canonicalization also performs a deep structural/type sanity check.
        _canonical_json(result)
        return result

    @staticmethod
    def _default_fetch_json(url: str, timeout_seconds: float) -> object:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Ranges-PokerStudyBlueprint/1",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise BlueprintFetchError(f"HTTP request failed for {url}: {error}") from error
        if len(body) > MAX_RESPONSE_BYTES:
            raise BlueprintFetchError(
                f"response from {url} exceeds {MAX_RESPONSE_BYTES} bytes"
            )
        return _strict_json_loads(body, url)

    def _read_cache(self, path: Path) -> tuple[object, str]:
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise BlueprintCacheError(f"missing cache file {path}") from error
        except OSError as error:
            raise BlueprintCacheError(f"cannot read cache file {path}: {error}") from error
        envelope = _strict_json_loads(raw, str(path))
        source = _require_mapping(envelope, f"cache envelope {path}")
        if set(source) != {"payload", "sha256"}:
            raise BlueprintCacheError(
                f"cache envelope {path} must contain only payload and sha256"
            )
        digest = source.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise BlueprintCacheError(f"cache envelope {path} has invalid sha256")
        payload = source.get("payload")
        actual_digest = canonical_response_sha256(payload)
        if actual_digest != digest:
            raise BlueprintCacheError(
                f"cache checksum mismatch for {path}: expected {digest}, got {actual_digest}"
            )
        return payload, digest

    def _write_cache(self, path: Path, payload: object) -> str:
        digest = canonical_response_sha256(payload)
        encoded = (
            _canonical_json({"payload": payload, "sha256": digest}) + "\n"
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return digest


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize and validate PokerStudy NL v2 preflop nodes"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="download exact nodes into local cache")
    sync.add_argument("--cache-dir", type=Path, default=Path("preflop_blueprint_cache"))
    sync.add_argument("--stack", required=True, type=_positive_int)
    sync.add_argument("--max-depth", type=_positive_int, default=2)
    sync.add_argument("--workers", type=_positive_int, default=4)
    sync.add_argument("--timeout-seconds", type=_positive_float, default=60.0)

    validate = subparsers.add_parser("validate", help="validate cache without network")
    validate.add_argument(
        "--cache-dir", type=Path, default=Path("preflop_blueprint_cache")
    )
    validate.add_argument("--stack", type=_positive_int)
    return parser


def _sync_command(
    args: argparse.Namespace,
    fetch_json: FetchJson | None,
) -> dict[str, int]:
    store = PokerStudyBlueprintStore(
        args.cache_dir,
        allow_network=True,
        timeout_seconds=args.timeout_seconds,
        fetch_json=fetch_json,
    )
    store.manifest()
    spots = store.spots(args.stack)
    selected = tuple(spot for spot in spots if spot.depth <= args.max_depth)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(store.node, args.stack, spot.history): spot.history
            for spot in selected
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
    return {
        "stack": args.stack,
        "max_depth": args.max_depth,
        "available_spots": len(spots),
        "synced_nodes": completed,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    fetch_json: FetchJson | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    try:
        if args.command == "sync":
            summary = _sync_command(args, fetch_json)
        else:
            store = PokerStudyBlueprintStore(args.cache_dir, allow_network=False)
            summary = store.validate_cache(args.stack)
    except BlueprintError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
