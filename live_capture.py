"""Provider-neutral continuous capture and keyframe selection primitives.

This module deliberately performs no OCR, model calls, poker-state inference,
or filesystem writes.  A caller supplies an image source and consumes selected
keyframes from an acknowledgement-aware bounded queue.

The queue never silently evicts an unacknowledged keyframe.  If capacity is
exhausted, the newly offered keyframe is rejected and a :class:`CaptureGap` is
recorded.  Downstream hand reconstruction can therefore fail closed instead of
assuming that the visual stream remained contiguous.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image


class FrameSource(Protocol):
    """Capture one atomic set of named image regions."""

    def capture(self) -> Mapping[str, Image.Image]:
        """Return a non-empty mapping of region name to image."""


class CaptureClock(Protocol):
    """Clock abstraction used to make capture scheduling deterministic in tests."""

    def monotonic_ns(self) -> int:
        """Return monotonic nanoseconds."""

    def utc_now(self) -> datetime:
        """Return an aware UTC timestamp."""

    def wait(self, event: threading.Event, timeout_seconds: float) -> bool:
        """Wait for ``event`` and return whether capture should stop."""


class SystemCaptureClock:
    """Production clock backed by :mod:`time` and ``Event.wait``."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def wait(self, event: threading.Event, timeout_seconds: float) -> bool:
        return event.wait(max(0.0, timeout_seconds))


@dataclass(frozen=True, slots=True)
class RegionSignature:
    """Small grayscale and edge samples for one named capture region."""

    name: str
    pixels: bytes
    edges: bytes

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("region signature name cannot be empty")
        if not self.pixels or len(self.pixels) != len(self.edges):
            raise ValueError("region signature samples must be non-empty and equal")


@dataclass(frozen=True, slots=True)
class RegionChange:
    """Measured visual change between two signatures for one region."""

    name: str
    pixel_fraction: float
    edge_fraction: float

    @property
    def score(self) -> float:
        return max(self.pixel_fraction, self.edge_fraction)


@dataclass(frozen=True, slots=True)
class FrameSignature:
    """Compact deterministic signature of an atomic named-region capture."""

    regions: tuple[RegionSignature, ...]
    sample_size: tuple[int, int]
    digest: str

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("a frame signature requires at least one region")
        if self.sample_size[0] <= 0 or self.sample_size[1] <= 0:
            raise ValueError("sample_size dimensions must be positive")
        names = [region.name for region in self.regions]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("frame signature regions must be unique and sorted")

    @classmethod
    def from_regions(
        cls,
        regions: Mapping[str, Image.Image],
        *,
        sample_size: tuple[int, int] = (48, 32),
        edge_delta: int = 18,
    ) -> "FrameSignature":
        """Create a compact local-CV signature without interpreting any text."""

        if not regions:
            raise ValueError("capture regions cannot be empty")
        width, height = sample_size
        if width <= 0 or height <= 0:
            raise ValueError("sample_size dimensions must be positive")
        if not 1 <= edge_delta <= 255:
            raise ValueError("edge_delta must be between 1 and 255")

        signatures: list[RegionSignature] = []
        digest = hashlib.sha256()
        digest.update(f"{width}x{height}:{edge_delta}".encode("ascii"))
        for name in sorted(regions):
            image = regions[name]
            if not isinstance(name, str) or not name:
                raise ValueError("capture region names must be non-empty strings")
            if not isinstance(image, Image.Image):
                raise TypeError(f"capture region {name!r} must be a PIL image")

            gray = image.convert("L").resize(
                (width, height),
                Image.Resampling.BILINEAR,
            )
            pixels = np.asarray(gray, dtype=np.uint8)
            horizontal = np.zeros_like(pixels, dtype=np.int16)
            vertical = np.zeros_like(pixels, dtype=np.int16)
            horizontal[:, 1:] = np.abs(
                pixels[:, 1:].astype(np.int16)
                - pixels[:, :-1].astype(np.int16)
            )
            vertical[1:, :] = np.abs(
                pixels[1:, :].astype(np.int16)
                - pixels[:-1, :].astype(np.int16)
            )
            edges = np.maximum(horizontal, vertical) >= edge_delta
            pixel_bytes = pixels.tobytes()
            edge_bytes = edges.astype(np.uint8).tobytes()
            signatures.append(
                RegionSignature(
                    name=name,
                    pixels=pixel_bytes,
                    edges=edge_bytes,
                )
            )
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(pixel_bytes)
            digest.update(edge_bytes)

        return cls(
            regions=tuple(signatures),
            sample_size=(width, height),
            digest=digest.hexdigest(),
        )

    def compare(
        self,
        other: "FrameSignature",
        *,
        pixel_delta: int = 20,
    ) -> tuple[RegionChange, ...]:
        """Measure changed-pixel and changed-edge fractions by region."""

        if not isinstance(other, FrameSignature):
            raise TypeError("other must be a FrameSignature")
        if not 1 <= pixel_delta <= 255:
            raise ValueError("pixel_delta must be between 1 and 255")

        left = {region.name: region for region in self.regions}
        right = {region.name: region for region in other.regions}
        changes: list[RegionChange] = []
        for name in sorted(set(left) | set(right)):
            first = left.get(name)
            second = right.get(name)
            if (
                first is None
                or second is None
                or len(first.pixels) != len(second.pixels)
                or self.sample_size != other.sample_size
            ):
                changes.append(RegionChange(name, 1.0, 1.0))
                continue

            first_pixels = np.frombuffer(first.pixels, dtype=np.uint8).astype(
                np.int16
            )
            second_pixels = np.frombuffer(second.pixels, dtype=np.uint8).astype(
                np.int16
            )
            first_edges = np.frombuffer(first.edges, dtype=np.uint8)
            second_edges = np.frombuffer(second.edges, dtype=np.uint8)
            changes.append(
                RegionChange(
                    name=name,
                    pixel_fraction=float(
                        np.mean(np.abs(first_pixels - second_pixels) >= pixel_delta)
                    ),
                    edge_fraction=float(np.mean(first_edges != second_edges)),
                )
            )
        return tuple(changes)


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    """One atomic capture with ordering, timestamps, images, and signature."""

    frame_id: int
    monotonic_ns: int
    captured_at: datetime
    regions: Mapping[str, Image.Image]
    signature: FrameSignature

    def __post_init__(self) -> None:
        if self.frame_id <= 0:
            raise ValueError("frame_id must be positive")
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns cannot be negative")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        if not self.regions:
            raise ValueError("capture frame regions cannot be empty")
        if set(self.regions) != {
            region.name for region in self.signature.regions
        }:
            raise ValueError("capture regions and signature regions must match")


class KeyframeReason(str, Enum):
    BASELINE = "baseline"
    CHANGE = "change"
    PEAK = "peak"
    STABLE = "stable"
    HEARTBEAT = "heartbeat"
    FLUSH = "flush"


@dataclass(frozen=True, slots=True)
class Keyframe:
    """A selected frame plus local, non-semantic selection evidence."""

    frame: CaptureFrame
    reason: KeyframeReason
    changed_regions: tuple[str, ...] = ()
    change_score: float = 0.0

    @property
    def frame_id(self) -> int:
        return self.frame.frame_id


class CaptureGapKind(str, Enum):
    QUEUE_OVERFLOW = "queue_overflow"
    CAPTURE_ERROR = "capture_error"


@dataclass(frozen=True, slots=True)
class CaptureGap:
    """Explicit evidence that one or more frames could not be retained."""

    kind: CaptureGapKind
    first_frame_id: int
    last_frame_id: int
    detected_monotonic_ns: int
    reason: str
    dropped_count: int = 1

    def __post_init__(self) -> None:
        if self.first_frame_id <= 0:
            raise ValueError("first_frame_id must be positive")
        if self.last_frame_id < self.first_frame_id:
            raise ValueError("last_frame_id cannot precede first_frame_id")
        if self.detected_monotonic_ns < 0:
            raise ValueError("detected_monotonic_ns cannot be negative")
        if not self.reason:
            raise ValueError("capture gap reason cannot be empty")
        if self.dropped_count <= 0:
            raise ValueError("dropped_count must be positive")


@dataclass(frozen=True, slots=True)
class GapSnapshot:
    """Atomic gap history plus its monotonic consumption revision."""

    revision: int
    gaps: tuple[CaptureGap, ...]


@dataclass(frozen=True, slots=True)
class CaptureGapEvent:
    """One immutable gap-ledger entry.

    ``GapSnapshot.gaps`` is intentionally coalesced for diagnostics.  Consumers
    need the immutable event stream as well: a coalesced range may grow after a
    consumer has seen its earlier form.
    """

    revision: int
    gap: CaptureGap

    def __post_init__(self) -> None:
        if self.revision <= 0:
            raise ValueError("gap event revision must be positive")


@dataclass(frozen=True, slots=True)
class KeyframeConsumerBatch:
    """Atomic keyframe/gap snapshot used by a transactional consumer."""

    acknowledged_through: int
    gap_revision: int
    keyframes: tuple[Keyframe, ...]
    gap_events: tuple[CaptureGapEvent, ...]


@dataclass(frozen=True, slots=True)
class CaughtUpRingRead:
    """One callback result bound to an atomic ring-ledger freshness check."""

    caught_up: bool
    reason: str
    consumed_through_frame_id: int
    observed_gap_revision: int
    acknowledged_through_frame_id: int
    current_gap_revision: int
    value: object | None = None


class KeyframeDetector:
    """Select baseline, transition-burst, stable, and heartbeat keyframes.

    The detector retains the first changed frame immediately.  While the visual
    scene is moving, it tracks the frame with the largest change from the
    previous stable anchor.  Once ``stable_frames`` consecutive frame pairs are
    stable, it emits the peak and final stable frame.  This keeps transition
    evidence without interpreting its poker meaning.
    """

    def __init__(
        self,
        *,
        pixel_fraction_threshold: float = 0.015,
        edge_fraction_threshold: float = 0.015,
        pixel_delta: int = 20,
        stable_frames: int = 2,
        heartbeat_interval_ns: int | None = 1_000_000_000,
        ignored_regions: Sequence[str] = (),
    ) -> None:
        for label, value in (
            ("pixel_fraction_threshold", pixel_fraction_threshold),
            ("edge_fraction_threshold", edge_fraction_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between 0 and 1")
        if not 1 <= pixel_delta <= 255:
            raise ValueError("pixel_delta must be between 1 and 255")
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        if heartbeat_interval_ns is not None and heartbeat_interval_ns <= 0:
            raise ValueError("heartbeat_interval_ns must be positive or None")

        self.pixel_fraction_threshold = pixel_fraction_threshold
        self.edge_fraction_threshold = edge_fraction_threshold
        self.pixel_delta = pixel_delta
        self.stable_frames = stable_frames
        self.heartbeat_interval_ns = heartbeat_interval_ns
        self.ignored_regions = frozenset(ignored_regions)

        self._anchor: CaptureFrame | None = None
        self._previous: CaptureFrame | None = None
        self._last_emitted_ns: int | None = None
        self._burst_first_id: int | None = None
        self._peak: CaptureFrame | None = None
        self._peak_regions: tuple[str, ...] = ()
        self._peak_score = 0.0
        self._stable_run = 0

    def _material_changes(
        self,
        before: FrameSignature,
        after: FrameSignature,
    ) -> tuple[tuple[str, ...], float]:
        changes = before.compare(after, pixel_delta=self.pixel_delta)
        material = tuple(
            change.name
            for change in changes
            if change.name not in self.ignored_regions
            and (
                (
                    change.pixel_fraction > 0
                    and change.pixel_fraction
                    >= self.pixel_fraction_threshold
                )
                or (
                    change.edge_fraction > 0
                    and change.edge_fraction
                    >= self.edge_fraction_threshold
                )
            )
        )
        score = max(
            (
                change.score
                for change in changes
                if change.name not in self.ignored_regions
            ),
            default=0.0,
        )
        return material, score

    def _emit(
        self,
        frame: CaptureFrame,
        reason: KeyframeReason,
        changed_regions: tuple[str, ...] = (),
        score: float = 0.0,
    ) -> Keyframe:
        self._last_emitted_ns = frame.monotonic_ns
        return Keyframe(
            frame=frame,
            reason=reason,
            changed_regions=changed_regions,
            change_score=score,
        )

    def observe(self, frame: CaptureFrame) -> tuple[Keyframe, ...]:
        """Consume one ordered frame and return newly selected keyframes."""

        if not isinstance(frame, CaptureFrame):
            raise TypeError("frame must be a CaptureFrame")
        if self._previous is not None:
            if frame.frame_id <= self._previous.frame_id:
                raise ValueError("capture frames must have increasing frame IDs")
            if frame.monotonic_ns < self._previous.monotonic_ns:
                raise ValueError("capture frame monotonic time cannot move backwards")

        if self._anchor is None:
            self._anchor = frame
            self._previous = frame
            return (self._emit(frame, KeyframeReason.BASELINE),)

        assert self._previous is not None
        pair_regions, _ = self._material_changes(
            self._previous.signature,
            frame.signature,
        )
        anchor_regions, anchor_score = self._material_changes(
            self._anchor.signature,
            frame.signature,
        )
        selected: list[Keyframe] = []

        if self._burst_first_id is None:
            if pair_regions or anchor_regions:
                self._burst_first_id = frame.frame_id
                self._peak = frame
                self._peak_regions = anchor_regions or pair_regions
                self._peak_score = anchor_score
                self._stable_run = 0
                selected.append(
                    self._emit(
                        frame,
                        KeyframeReason.CHANGE,
                        anchor_regions or pair_regions,
                        anchor_score,
                    )
                )
            elif (
                self.heartbeat_interval_ns is not None
                and self._last_emitted_ns is not None
                and frame.monotonic_ns - self._last_emitted_ns
                >= self.heartbeat_interval_ns
            ):
                selected.append(self._emit(frame, KeyframeReason.HEARTBEAT))
        else:
            if anchor_score > self._peak_score:
                self._peak = frame
                self._peak_regions = anchor_regions
                self._peak_score = anchor_score

            if pair_regions:
                self._stable_run = 0
            else:
                self._stable_run += 1

            if self._stable_run >= self.stable_frames:
                assert self._peak is not None
                if self._peak.frame_id not in {
                    self._burst_first_id,
                    frame.frame_id,
                }:
                    selected.append(
                        self._emit(
                            self._peak,
                            KeyframeReason.PEAK,
                            self._peak_regions,
                            self._peak_score,
                        )
                    )
                if frame.frame_id != self._burst_first_id:
                    selected.append(
                        self._emit(
                            frame,
                            KeyframeReason.STABLE,
                            anchor_regions,
                            anchor_score,
                        )
                    )
                self._anchor = frame
                self._burst_first_id = None
                self._peak = None
                self._peak_regions = ()
                self._peak_score = 0.0
                self._stable_run = 0

        self._previous = frame
        return tuple(selected)

    def flush(self) -> tuple[Keyframe, ...]:
        """Return retained evidence for an unfinished transition burst."""

        if self._burst_first_id is None or self._previous is None:
            return ()
        selected: list[Keyframe] = []
        assert self._peak is not None
        if self._peak.frame_id != self._burst_first_id:
            selected.append(
                self._emit(
                    self._peak,
                    KeyframeReason.PEAK,
                    self._peak_regions,
                    self._peak_score,
                )
            )
        if self._previous.frame_id not in {
            self._burst_first_id,
            self._peak.frame_id,
        }:
            selected.append(
                self._emit(
                    self._previous,
                    KeyframeReason.FLUSH,
                    self._peak_regions,
                    self._peak_score,
                )
            )
        self._anchor = self._previous
        self._burst_first_id = None
        self._peak = None
        self._peak_regions = ()
        self._peak_score = 0.0
        self._stable_run = 0
        return tuple(selected)


class KeyframeRing:
    """Thread-safe bounded keyframe storage with an ACK watermark.

    Acknowledged entries may be discarded to make room.  An unacknowledged
    entry is never silently evicted: if no safe slot exists, the new keyframe is
    rejected and a queue-overflow :class:`CaptureGap` is recorded.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: deque[Keyframe] = deque()
        self._gaps: list[CaptureGap] = []
        self._gap_events: list[CaptureGapEvent] = []
        self._gap_revision = 0
        self._acked_through = 0
        self._last_offered_frame_id = 0
        self._condition = threading.Condition()

    @property
    def acknowledged_through(self) -> int:
        with self._condition:
            return self._acked_through

    @property
    def gap_revision(self) -> int:
        """Monotonic revision incremented for every recorded gap event."""

        with self._condition:
            return self._gap_revision

    def _coalesce_gap_locked(self, gap: CaptureGap) -> CaptureGap:
        if self._gaps:
            previous = self._gaps[-1]
            if (
                previous.kind == gap.kind
                and previous.reason == gap.reason
                and gap.first_frame_id >= previous.first_frame_id
                and gap.first_frame_id <= previous.last_frame_id + 1
            ):
                combined = replace(
                    previous,
                    last_frame_id=max(
                        previous.last_frame_id,
                        gap.last_frame_id,
                    ),
                    dropped_count=previous.dropped_count + gap.dropped_count,
                )
                self._gaps[-1] = combined
                self._gap_revision += 1
                self._gap_events.append(
                    CaptureGapEvent(self._gap_revision, gap)
                )
                self._condition.notify_all()
                return combined
        self._gaps.append(gap)
        self._gaps.sort(
            key=lambda item: (item.first_frame_id, item.last_frame_id)
        )
        self._gap_revision += 1
        self._gap_events.append(CaptureGapEvent(self._gap_revision, gap))
        self._condition.notify_all()
        return gap

    def record_gap(self, gap: CaptureGap) -> CaptureGap:
        """Record an externally detected capture gap."""

        if not isinstance(gap, CaptureGap):
            raise TypeError("gap must be a CaptureGap")
        with self._condition:
            return self._coalesce_gap_locked(gap)

    def offer(self, keyframe: Keyframe) -> CaptureGap | None:
        """Store a keyframe or return an explicit overflow gap."""

        if not isinstance(keyframe, Keyframe):
            raise TypeError("keyframe must be a Keyframe")
        with self._condition:
            if keyframe.frame_id <= self._last_offered_frame_id:
                raise ValueError("offered keyframes must have increasing frame IDs")
            self._last_offered_frame_id = keyframe.frame_id
            if keyframe.frame_id <= self._acked_through:
                return None

            while (
                len(self._items) >= self.capacity
                and self._items
                and self._items[0].frame_id <= self._acked_through
            ):
                self._items.popleft()

            if len(self._items) >= self.capacity:
                gap = CaptureGap(
                    kind=CaptureGapKind.QUEUE_OVERFLOW,
                    first_frame_id=keyframe.frame_id,
                    last_frame_id=keyframe.frame_id,
                    detected_monotonic_ns=keyframe.frame.monotonic_ns,
                    reason="unacknowledged keyframe capacity exceeded",
                )
                return self._coalesce_gap_locked(gap)

            self._items.append(keyframe)
            self._condition.notify_all()
            return None

    def acknowledge(self, through_frame_id: int) -> None:
        """Advance the ACK watermark and release retained frames at/below it."""

        if through_frame_id < 0:
            raise ValueError("through_frame_id cannot be negative")
        with self._condition:
            if through_frame_id < self._acked_through:
                raise ValueError("acknowledgement watermark cannot move backwards")
            self._acked_through = through_frame_id
            while self._items and self._items[0].frame_id <= through_frame_id:
                self._items.popleft()
            self._condition.notify_all()

    def pending(self, *, after_frame_id: int = 0) -> tuple[Keyframe, ...]:
        """Return an immutable ordered snapshot of pending keyframes."""

        with self._condition:
            return tuple(
                keyframe
                for keyframe in self._items
                if keyframe.frame_id > after_frame_id
            )

    def gaps(self) -> tuple[CaptureGap, ...]:
        with self._condition:
            return tuple(self._gaps)

    def gap_snapshot(self) -> GapSnapshot:
        """Atomically return the gap revision and gap history.

        Consumers should process this snapshot, then pass ``snapshot.revision``
        to :meth:`wait_for_pending`.  Reading ``gaps`` and ``gap_revision``
        separately can otherwise race with a newly recorded gap.
        """

        with self._condition:
            return GapSnapshot(
                revision=self._gap_revision,
                gaps=tuple(self._gaps),
            )

    def consumer_batch(
        self,
        *,
        after_frame_id: int = 0,
        after_gap_revision: int = 0,
    ) -> KeyframeConsumerBatch:
        """Return one atomic ledger snapshot for an ordered consumer.

        Gap entries are immutable raw events rather than the coalesced
        diagnostic ranges returned by :meth:`gaps`.  A consumer can therefore
        remember ``gap_revision`` without missing a later extension of an
        already visible range.
        """

        if after_frame_id < 0:
            raise ValueError("after_frame_id cannot be negative")
        with self._condition:
            if (
                after_gap_revision < 0
                or after_gap_revision > self._gap_revision
            ):
                raise ValueError(
                    "after_gap_revision is outside the gap ledger"
                )
            return KeyframeConsumerBatch(
                acknowledged_through=self._acked_through,
                gap_revision=self._gap_revision,
                keyframes=tuple(
                    item
                    for item in self._items
                    if item.frame_id > after_frame_id
                ),
                gap_events=tuple(
                    event
                    for event in self._gap_events
                    if event.revision > after_gap_revision
                ),
            )

    def read_if_caught_up(
        self,
        *,
        consumed_through_frame_id: int,
        observed_gap_revision: int,
        reader: Callable[[], object],
    ) -> CaughtUpRingRead:
        """Run ``reader`` only while the consumer exactly covers this ledger.

        The callback executes while the ring condition is held.  A caller that
        also reads downstream state must acquire its locks before this method
        in the worker's lock order (worker, coordinator, ring).  A producer
        therefore cannot publish a keyframe or gap between this check and the
        accepted-state read.
        """

        if consumed_through_frame_id < 0:
            raise ValueError(
                "consumed_through_frame_id cannot be negative"
            )
        if observed_gap_revision < 0:
            raise ValueError("observed_gap_revision cannot be negative")
        if not callable(reader):
            raise TypeError("reader must be callable")
        with self._condition:
            if observed_gap_revision > self._gap_revision:
                raise ValueError(
                    "observed_gap_revision is outside the gap ledger"
                )

            def result(
                caught_up: bool,
                reason: str,
                value: object | None = None,
            ) -> CaughtUpRingRead:
                return CaughtUpRingRead(
                    caught_up=caught_up,
                    reason=reason,
                    consumed_through_frame_id=(
                        consumed_through_frame_id
                    ),
                    observed_gap_revision=observed_gap_revision,
                    acknowledged_through_frame_id=self._acked_through,
                    current_gap_revision=self._gap_revision,
                    value=value,
                )

            if self._gap_revision > observed_gap_revision:
                return result(
                    False,
                    "capture gap ledger has an unseen revision "
                    f"{self._gap_revision} (worker observed "
                    f"{observed_gap_revision})",
                )
            pending = tuple(
                item.frame_id
                for item in self._items
                if item.frame_id > consumed_through_frame_id
            )
            if pending:
                return result(
                    False,
                    "pending keyframe "
                    f"{pending[0]} has not been consumed "
                    f"(worker through {consumed_through_frame_id})",
                )
            if self._acked_through != consumed_through_frame_id:
                return result(
                    False,
                    "ring ACK differs from worker consumption "
                    f"(ACK {self._acked_through}, worker through "
                    f"{consumed_through_frame_id})",
                )
            if self._last_offered_frame_id > consumed_through_frame_id:
                return result(
                    False,
                    "offered frame watermark is newer than worker "
                    f"consumption ({self._last_offered_frame_id} > "
                    f"{consumed_through_frame_id})",
                )
            return result(True, "", reader())

    def acknowledge_consumer_batch(
        self,
        through_frame_id: int,
        *,
        expected_gap_revision: int,
    ) -> bool:
        """ACK only if no gap appeared after a consumer's atomic snapshot.

        Returning ``False`` leaves both the ACK watermark and retained
        keyframes untouched.  The consumer must read the newer gap ledger,
        invalidate downstream state, and retry.
        """

        if through_frame_id < 0:
            raise ValueError("through_frame_id cannot be negative")
        with self._condition:
            if (
                expected_gap_revision < 0
                or expected_gap_revision > self._gap_revision
            ):
                raise ValueError(
                    "expected_gap_revision is outside the gap ledger"
                )
            if expected_gap_revision != self._gap_revision:
                return False
            if through_frame_id < self._acked_through:
                raise ValueError("acknowledgement watermark cannot move backwards")
            self._acked_through = through_frame_id
            while (
                self._items
                and self._items[0].frame_id <= through_frame_id
            ):
                self._items.popleft()
            self._condition.notify_all()
            return True

    def wait_for_pending(
        self,
        *,
        after_frame_id: int = 0,
        after_gap_revision: int | None = None,
        timeout_seconds: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> tuple[Keyframe, ...]:
        """Wait for a newer keyframe or an unseen gap, then return pending.

        ``after_gap_revision`` is the caller's gap cursor.  When omitted, gaps
        already present at call time are treated as seen; this prevents one
        historical gap from turning every later wait into a hot loop.
        """

        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        if stop_event is not None and not isinstance(
            stop_event,
            threading.Event,
        ):
            raise TypeError("stop_event must be a threading.Event or None")
        with self._condition:
            gap_cursor = (
                self._gap_revision
                if after_gap_revision is None
                else after_gap_revision
            )
            if gap_cursor < 0 or gap_cursor > self._gap_revision:
                raise ValueError(
                    "after_gap_revision is outside the gap history"
                )
            self._condition.wait_for(
                lambda: any(
                    item.frame_id > after_frame_id for item in self._items
                )
                or self._gap_revision > gap_cursor
                or (stop_event is not None and stop_event.is_set()),
                timeout=timeout_seconds,
            )
            return tuple(
                keyframe
                for keyframe in self._items
                if keyframe.frame_id > after_frame_id
            )

    def wake_consumers(self) -> None:
        """Wake condition waiters so they can observe external stop state."""

        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class CaptureCycle:
    """Result of one synchronous service capture attempt."""

    frame: CaptureFrame | None
    keyframes: tuple[Keyframe, ...] = ()
    gaps: tuple[CaptureGap, ...] = ()


class ContinuousCaptureService:
    """Sample an injected source and publish locally selected keyframes."""

    def __init__(
        self,
        source: FrameSource | Callable[[], Mapping[str, Image.Image]],
        *,
        detector: KeyframeDetector | None = None,
        ring: KeyframeRing | None = None,
        clock: CaptureClock | None = None,
        fps: float = 8.0,
        signature_factory: Callable[
            [Mapping[str, Image.Image]], FrameSignature
        ] = FrameSignature.from_regions,
    ) -> None:
        if not callable(source) and not callable(getattr(source, "capture", None)):
            raise TypeError("source must be callable or implement capture()")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        if not callable(signature_factory):
            raise TypeError("signature_factory must be callable")

        self.source = source
        self.detector = detector or KeyframeDetector()
        self.ring = ring or KeyframeRing(256)
        self.clock = clock or SystemCaptureClock()
        self.fps = float(fps)
        self.signature_factory = signature_factory

        self._next_frame_id = 1
        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._started = False
        self._closed = False
        self._sync_owner_thread_id: int | None = None
        self._background_error: BaseException | None = None

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def background_error(self) -> BaseException | None:
        """Return the terminal sampler/cleanup failure, if one occurred."""

        with self._state_lock:
            return self._background_error

    def _source_capture(self) -> Mapping[str, Image.Image]:
        capture_method = getattr(self.source, "capture", None)
        raw = capture_method() if callable(capture_method) else self.source()
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("capture source must return a non-empty mapping")
        copied: dict[str, Image.Image] = {}
        for name, image in raw.items():
            if not isinstance(name, str) or not name:
                raise ValueError("capture region names must be non-empty strings")
            if not isinstance(image, Image.Image):
                raise TypeError(f"capture region {name!r} must be a PIL image")
            copied[name] = image.copy()
        return MappingProxyType(copied)

    def _offer(self, keyframes: Sequence[Keyframe]) -> tuple[CaptureGap, ...]:
        gaps = []
        for keyframe in keyframes:
            gap = self.ring.offer(keyframe)
            if gap is not None:
                gaps.append(gap)
        return tuple(gaps)

    def capture_once(self) -> CaptureCycle:
        """Capture and process one frame synchronously."""

        caller_thread_id = threading.get_ident()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("capture service is closed")
            if self._started:
                if threading.current_thread() is not self._thread:
                    raise RuntimeError(
                        "synchronous capture cannot run on a background service"
                    )
            elif self._sync_owner_thread_id is None:
                self._sync_owner_thread_id = caller_thread_id
            elif self._sync_owner_thread_id != caller_thread_id:
                raise RuntimeError(
                    "synchronous capture cannot cross threads"
                )
        with self._capture_lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            monotonic_ns = self.clock.monotonic_ns()
            captured_at = self.clock.utc_now()
            try:
                regions = self._source_capture()
                signature = self.signature_factory(regions)
                frame = CaptureFrame(
                    frame_id=frame_id,
                    monotonic_ns=monotonic_ns,
                    captured_at=captured_at,
                    regions=regions,
                    signature=signature,
                )
                keyframes = self.detector.observe(frame)
                return CaptureCycle(
                    frame=frame,
                    keyframes=keyframes,
                    gaps=self._offer(keyframes),
                )
            except Exception as error:
                gap = CaptureGap(
                    kind=CaptureGapKind.CAPTURE_ERROR,
                    first_frame_id=frame_id,
                    last_frame_id=frame_id,
                    detected_monotonic_ns=monotonic_ns,
                    reason=f"capture failed: {type(error).__name__}",
                )
                recorded = self.ring.record_gap(gap)
                return CaptureCycle(frame=None, gaps=(recorded,))

    def _run(self) -> None:
        period_ns = max(1, round(1_000_000_000 / self.fps))
        next_deadline = self.clock.monotonic_ns()
        with self._state_lock:
            self._running = True
        try:
            while not self._stop_event.is_set():
                self.capture_once()
                now = self.clock.monotonic_ns()
                next_deadline += period_ns
                if next_deadline <= now:
                    next_deadline = now + period_ns
                if self.clock.wait(
                    self._stop_event,
                    (next_deadline - now) / 1_000_000_000,
                ):
                    break
        except BaseException as error:
            with self._state_lock:
                self._background_error = error
        finally:
            try:
                with self._capture_lock:
                    self._offer(self.detector.flush())
                    close_source = getattr(self.source, "close", None)
                    if callable(close_source):
                        close_source()
            except BaseException as error:
                with self._state_lock:
                    if self._background_error is None:
                        self._background_error = error
            finally:
                with self._state_lock:
                    self._running = False
                    self._closed = True

    def start(self, *, thread_name: str = "continuous-table-capture") -> None:
        """Start the background sampler once."""

        with self._state_lock:
            if self._started:
                raise RuntimeError("capture service cannot be restarted")
            if self._closed:
                raise RuntimeError("capture service is closed")
            if self._sync_owner_thread_id is not None:
                raise RuntimeError(
                    "background capture cannot start after synchronous capture"
                )
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=thread_name,
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """Request shutdown and wait for the sampler thread."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            with self._state_lock:
                if self._closed:
                    return
                owner = self._sync_owner_thread_id
            if owner is not None and owner != threading.get_ident():
                raise RuntimeError(
                    "synchronous capture service must close on its owner thread"
                )
            with self._capture_lock:
                self._offer(self.detector.flush())
                close_source = getattr(self.source, "close", None)
                if callable(close_source):
                    close_source()
            with self._state_lock:
                self._closed = True
            return
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise TimeoutError("capture service did not stop before timeout")
        with self._state_lock:
            background_error = self._background_error
        if background_error is not None:
            raise RuntimeError(
                "capture service background cleanup failed"
            ) from background_error


__all__ = [
    "CaughtUpRingRead",
    "CaptureClock",
    "CaptureCycle",
    "CaptureFrame",
    "CaptureGap",
    "CaptureGapEvent",
    "CaptureGapKind",
    "ContinuousCaptureService",
    "FrameSignature",
    "FrameSource",
    "GapSnapshot",
    "Keyframe",
    "KeyframeConsumerBatch",
    "KeyframeDetector",
    "KeyframeReason",
    "KeyframeRing",
    "RegionChange",
    "RegionSignature",
    "SystemCaptureClock",
]
