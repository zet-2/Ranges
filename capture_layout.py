"""Versioned, fail-closed capture layouts for PokerStars table crops.

The live vision protocol uses six canonical seat slots with Hero fixed at S4.
A native heads-up table therefore still maps its two physical player pods to
S0 and S4; the four unused canonical slots are rendered as explicit blanks.

This module only validates profile metadata and coordinates.  It deliberately
does not infer regions from an unknown screenshot and does not mark a profile
as verified merely because its JSON is structurally valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "pokerstars_capture_layout"
CALIBRATION_METHOD = "interactive_region_corners"
CANONICAL_HERO_SEAT_INDEX = 4
CANONICAL_HU_OPPONENT_SEAT_INDEX = 0
CANONICAL_SEAT_CAPTURES = {
    "seat1": 0,
    "seat2": 1,
    "seat3": 2,
    "seat4": 3,
    "hero": 4,
    "seat6": 5,
}
SIX_MAX_REGION_NAMES = frozenset((*CANONICAL_SEAT_CAPTURES, "board", "buttons"))
HEADS_UP_REGION_NAMES = frozenset(("seat1", "hero", "board", "buttons"))

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CaptureLayoutError(ValueError):
    """Raised when a capture layout violates the strict profile contract."""


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaptureLayoutError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CaptureLayoutError(f"{field_name} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], field_name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CaptureLayoutError(
            f"{field_name} keys mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _positive_int(value: object, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureLayoutError(f"{field_name} must be an integer")
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise CaptureLayoutError(f"{field_name} must be {qualifier}")
    return value


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise CaptureLayoutError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern!r}"
        )
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CaptureLayoutError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "left", _positive_int(self.left, "region left", allow_zero=True)
        )
        object.__setattr__(
            self, "top", _positive_int(self.top, "region top", allow_zero=True)
        )
        object.__setattr__(self, "width", _positive_int(self.width, "region width"))
        object.__setattr__(
            self, "height", _positive_int(self.height, "region height")
        )

    def to_payload(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class CaptureCalibration:
    status: str
    reference_frame_sha256: str
    crop_sha256: tuple[tuple[str, str], ...]
    reviewed_reference_frame_sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in {
            "draft",
            "verified",
        }:
            raise CaptureLayoutError("calibration status must be draft or verified")
        reference = _sha256(
            self.reference_frame_sha256,
            "calibration reference_frame_sha256",
        )
        crops = tuple(sorted(self.crop_sha256))
        names = [name for name, _ in crops]
        if not crops or len(names) != len(set(names)):
            raise CaptureLayoutError(
                "calibration crop_sha256 names must be non-empty and unique"
            )
        for name, digest in crops:
            if not isinstance(name, str):
                raise CaptureLayoutError("calibration crop names must be strings")
            _sha256(digest, f"calibration crop_sha256[{name!r}]")

        reviewed = self.reviewed_reference_frame_sha256
        if self.status == "draft":
            if reviewed is not None:
                raise CaptureLayoutError(
                    "draft calibration cannot have a reviewed frame digest"
                )
        else:
            reviewed = _sha256(
                reviewed,
                "calibration reviewed_reference_frame_sha256",
            )
            if reviewed != reference:
                raise CaptureLayoutError(
                    "verified calibration must review the exact reference frame"
                )

        object.__setattr__(self, "reference_frame_sha256", reference)
        object.__setattr__(self, "crop_sha256", crops)
        object.__setattr__(self, "reviewed_reference_frame_sha256", reviewed)

    def to_payload(self) -> dict[str, object]:
        return {
            "method": CALIBRATION_METHOD,
            "status": self.status,
            "reference_frame_sha256": self.reference_frame_sha256,
            "crop_sha256": dict(self.crop_sha256),
            "reviewed_reference_frame_sha256": (
                self.reviewed_reference_frame_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class CaptureLayout:
    layout_id: str
    table_format: str
    monitor_index: int
    monitor_width_px: int
    monitor_height_px: int
    hero_seat_index: int
    opponent_seat_index: int | None
    regions: tuple[tuple[str, CaptureRegion], ...]
    calibration: CaptureCalibration

    def __post_init__(self) -> None:
        layout_id = _identifier(self.layout_id, "layout_id")
        if not isinstance(self.table_format, str) or self.table_format not in {
            "six_max",
            "heads_up",
        }:
            raise CaptureLayoutError("table_format must be six_max or heads_up")
        monitor_index = _positive_int(self.monitor_index, "monitor index")
        width = _positive_int(self.monitor_width_px, "monitor width_px")
        height = _positive_int(self.monitor_height_px, "monitor height_px")
        if self.hero_seat_index != CANONICAL_HERO_SEAT_INDEX:
            raise CaptureLayoutError("canonical Hero must remain S4")

        regions = tuple(sorted(self.regions))
        names = [name for name, _ in regions]
        if len(names) != len(set(names)):
            raise CaptureLayoutError("region names must be unique")
        if any(not isinstance(region, CaptureRegion) for _, region in regions):
            raise CaptureLayoutError("every region must be a CaptureRegion")

        if self.table_format == "heads_up":
            if self.opponent_seat_index != CANONICAL_HU_OPPONENT_SEAT_INDEX:
                raise CaptureLayoutError("native HU opponent must map to S0")
            expected_regions = HEADS_UP_REGION_NAMES
        else:
            if self.opponent_seat_index is not None:
                raise CaptureLayoutError(
                    "six_max canonical seats cannot declare one opponent index"
                )
            expected_regions = SIX_MAX_REGION_NAMES
        if set(names) != expected_regions:
            raise CaptureLayoutError(
                f"{self.table_format} regions must be {sorted(expected_regions)}"
            )

        coordinate_boxes = set()
        for name, region in regions:
            if region.left + region.width > width:
                raise CaptureLayoutError(
                    f"region {name!r} extends beyond monitor width"
                )
            if region.top + region.height > height:
                raise CaptureLayoutError(
                    f"region {name!r} extends beyond monitor height"
                )
            box = (region.left, region.top, region.width, region.height)
            if box in coordinate_boxes:
                raise CaptureLayoutError("two semantic regions cannot be identical")
            coordinate_boxes.add(box)

        crop_names = {name for name, _ in self.calibration.crop_sha256}
        if crop_names != set(names):
            raise CaptureLayoutError(
                "calibration crop hashes must exactly cover the declared regions"
            )

        object.__setattr__(self, "layout_id", layout_id)
        object.__setattr__(self, "monitor_index", monitor_index)
        object.__setattr__(self, "monitor_width_px", width)
        object.__setattr__(self, "monitor_height_px", height)
        object.__setattr__(self, "regions", regions)

    @property
    def region_map(self) -> dict[str, dict[str, int]]:
        return {name: region.to_payload() for name, region in self.regions}

    @property
    def active_seat_indices(self) -> frozenset[int]:
        if self.table_format == "heads_up":
            return frozenset(
                (CANONICAL_HU_OPPONENT_SEAT_INDEX, CANONICAL_HERO_SEAT_INDEX)
            )
        return frozenset(range(6))

    @property
    def opponent_capture_seats(self) -> dict[str, int]:
        return {
            name: index
            for name, index in CANONICAL_SEAT_CAPTURES.items()
            if index != CANONICAL_HERO_SEAT_INDEX and name in self.region_map
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "layout_id": self.layout_id,
            "table_format": self.table_format,
            "monitor": {
                "index": self.monitor_index,
                "width_px": self.monitor_width_px,
                "height_px": self.monitor_height_px,
            },
            "canonical_seats": {
                "hero": self.hero_seat_index,
                "opponent": self.opponent_seat_index,
            },
            "regions": self.region_map,
            "calibration": self.calibration.to_payload(),
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def require_verified(self) -> None:
        if self.calibration.status != "verified":
            raise CaptureLayoutError(
                "capture layout is draft; review the hashed calibration evidence first"
            )

    def assert_monitor(self, index: int, width: int, height: int) -> None:
        if (index, width, height) != (
            self.monitor_index,
            self.monitor_width_px,
            self.monitor_height_px,
        ):
            raise CaptureLayoutError(
                "selected monitor changed: profile expects index "
                f"{self.monitor_index} at {self.monitor_width_px}x"
                f"{self.monitor_height_px}, got index {index} at {width}x{height}"
            )


def parse_capture_layout(payload: object) -> CaptureLayout:
    root = _mapping(payload, "capture layout")
    _exact_keys(
        root,
        {
            "schema_version",
            "artifact_type",
            "layout_id",
            "table_format",
            "monitor",
            "canonical_seats",
            "regions",
            "calibration",
        },
        "capture layout",
    )
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise CaptureLayoutError(
            f"unsupported capture layout schema_version={root['schema_version']!r}"
        )
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise CaptureLayoutError("unexpected capture layout artifact_type")

    monitor = _mapping(root["monitor"], "monitor")
    _exact_keys(monitor, {"index", "width_px", "height_px"}, "monitor")
    canonical = _mapping(root["canonical_seats"], "canonical_seats")
    _exact_keys(canonical, {"hero", "opponent"}, "canonical_seats")

    raw_regions = _mapping(root["regions"], "regions")
    regions: list[tuple[str, CaptureRegion]] = []
    for name, raw_region in raw_regions.items():
        values = _mapping(raw_region, f"regions[{name!r}]")
        _exact_keys(values, {"left", "top", "width", "height"}, f"regions[{name!r}]")
        regions.append(
            (
                name,
                CaptureRegion(
                    left=values["left"],
                    top=values["top"],
                    width=values["width"],
                    height=values["height"],
                ),
            )
        )

    raw_calibration = _mapping(root["calibration"], "calibration")
    _exact_keys(
        raw_calibration,
        {
            "method",
            "status",
            "reference_frame_sha256",
            "crop_sha256",
            "reviewed_reference_frame_sha256",
        },
        "calibration",
    )
    if raw_calibration["method"] != CALIBRATION_METHOD:
        raise CaptureLayoutError("unsupported calibration method")
    crop_hashes = _mapping(raw_calibration["crop_sha256"], "calibration crop_sha256")
    calibration = CaptureCalibration(
        status=raw_calibration["status"],
        reference_frame_sha256=raw_calibration["reference_frame_sha256"],
        crop_sha256=tuple(crop_hashes.items()),
        reviewed_reference_frame_sha256=raw_calibration[
            "reviewed_reference_frame_sha256"
        ],
    )

    return CaptureLayout(
        layout_id=root["layout_id"],
        table_format=root["table_format"],
        monitor_index=monitor["index"],
        monitor_width_px=monitor["width_px"],
        monitor_height_px=monitor["height_px"],
        hero_seat_index=canonical["hero"],
        opponent_seat_index=canonical["opponent"],
        regions=tuple(regions),
        calibration=calibration,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureLayoutError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_capture_layout(
    path: str | Path,
    *,
    require_verified: bool = True,
) -> CaptureLayout:
    artifact_path = Path(path)
    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CaptureLayoutError(f"invalid JSON constant {value}")
            ),
        )
    except CaptureLayoutError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CaptureLayoutError(
            f"cannot read capture layout {artifact_path}: {error}"
        ) from error
    layout = parse_capture_layout(payload)
    if require_verified:
        layout.require_verified()
    return layout


def verified_payload(layout: CaptureLayout) -> dict[str, object]:
    """Return the same profile marked reviewed against its exact frame hash."""
    if layout.calibration.status != "draft":
        raise CaptureLayoutError("only a draft layout can be approved")
    payload = layout.to_payload()
    payload["calibration"]["status"] = "verified"
    payload["calibration"]["reviewed_reference_frame_sha256"] = (
        layout.calibration.reference_frame_sha256
    )
    return parse_capture_layout(payload).to_payload()
