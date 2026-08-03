"""Persistent MSS source for the provider-neutral continuous capture service."""

from __future__ import annotations

import threading
from typing import Callable, Mapping

import mss
from PIL import Image


class MSSRegionFrameSource:
    """Atomically capture one calibrated union and crop named table regions.

    The MSS context is created lazily on the capture thread and reused for the
    service lifetime.  ``close`` must run on that same thread; the capture
    service does this from its worker ``finally`` block.
    """

    def __init__(
        self,
        monitor: int,
        zones: Mapping[str, Mapping[str, int]],
        *,
        mss_factory: Callable[[], object] = mss.MSS,
    ) -> None:
        if isinstance(monitor, bool) or not isinstance(monitor, int) or monitor < 1:
            raise ValueError("monitor must be a positive integer")
        if not zones:
            raise ValueError("zones cannot be empty")
        normalized: dict[str, dict[str, int]] = {}
        for name, zone in zones.items():
            if not isinstance(name, str) or not name or not isinstance(zone, Mapping):
                raise ValueError("zones must map non-empty names to rectangles")
            try:
                rectangle = {
                    key: int(zone[key])
                    for key in ("left", "top", "width", "height")
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"zone {name!r} must contain integer left/top/width/height"
                ) from error
            if rectangle["width"] <= 0 or rectangle["height"] <= 0:
                raise ValueError(f"zone {name!r} dimensions must be positive")
            normalized[name] = rectangle
        if not callable(mss_factory):
            raise TypeError("mss_factory must be callable")
        self.monitor = monitor
        self.zones = normalized
        self.mss_factory = mss_factory
        self._sct = None
        self._owner_thread_id: int | None = None
        self._state_lock = threading.Lock()

    def _context(self):
        thread_id = threading.get_ident()
        with self._state_lock:
            if (
                self._owner_thread_id is not None
                and self._owner_thread_id != thread_id
            ):
                raise RuntimeError(
                    "MSSRegionFrameSource cannot cross capture threads"
                )
            if self._sct is None:
                self._sct = self.mss_factory()
                self._owner_thread_id = thread_id
            return self._sct

    def capture(self) -> dict[str, Image.Image]:
        context = self._context()
        monitors = getattr(context, "monitors", None)
        if not isinstance(monitors, list) or self.monitor >= len(monitors):
            raise RuntimeError(f"monitor {self.monitor} is not available")
        selected = monitors[self.monitor]
        min_left = min(zone["left"] for zone in self.zones.values())
        min_top = min(zone["top"] for zone in self.zones.values())
        max_right = max(
            zone["left"] + zone["width"] for zone in self.zones.values()
        )
        max_bottom = max(
            zone["top"] + zone["height"] for zone in self.zones.values()
        )
        bounding_region = {
            "left": int(selected["left"]) + min_left,
            "top": int(selected["top"]) + min_top,
            "width": max_right - min_left,
            "height": max_bottom - min_top,
        }
        grabbed = context.grab(bounding_region)
        frame = Image.frombytes(
            "RGB",
            grabbed.size,
            grabbed.bgra,
            "raw",
            "BGRX",
        )
        captures: dict[str, Image.Image] = {}
        for name, zone in self.zones.items():
            left = zone["left"] - min_left
            top = zone["top"] - min_top
            captures[name] = frame.crop(
                (
                    left,
                    top,
                    left + zone["width"],
                    top + zone["height"],
                )
            )
        return captures

    def close(self) -> None:
        with self._state_lock:
            if self._sct is None:
                return
            if threading.get_ident() != self._owner_thread_id:
                raise RuntimeError(
                    "MSS context must close on its capture thread"
                )
            close = getattr(self._sct, "close", None)
            if callable(close):
                close()
            self._sct = None
            self._owner_thread_id = None


__all__ = ["MSSRegionFrameSource"]
