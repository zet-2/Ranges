"""Tests for the PokerStars MSS frame source."""

from __future__ import annotations

import threading
from types import SimpleNamespace
import unittest

from PIL import Image

from pokerstars_live_capture import MSSRegionFrameSource


class FakeMSS:
    def __init__(self):
        self.monitors = [
            {"left": 0, "top": 0, "width": 100, "height": 100},
            {"left": 100, "top": 200, "width": 100, "height": 80},
        ]
        self.grabs = []
        self.closed = False

    def grab(self, region):
        self.grabs.append(region)
        image = Image.new(
            "RGBA",
            (region["width"], region["height"]),
            (30, 20, 10, 255),
        )
        # PIL raw BGRX expects bytes ordered B,G,R,X.
        bgra = bytes((10, 20, 30, 255)) * (
            region["width"] * region["height"]
        )
        return SimpleNamespace(size=image.size, bgra=bgra)

    def close(self):
        self.closed = True


class MSSRegionFrameSourceTests(unittest.TestCase):
    def test_reuses_one_context_and_crops_one_atomic_union(self):
        contexts = []

        def factory():
            context = FakeMSS()
            contexts.append(context)
            return context

        source = MSSRegionFrameSource(
            1,
            {
                "seat": {"left": 10, "top": 5, "width": 20, "height": 10},
                "board": {
                    "left": 40,
                    "top": 20,
                    "width": 30,
                    "height": 15,
                },
            },
            mss_factory=factory,
        )

        first = source.capture()
        second = source.capture()

        self.assertEqual(1, len(contexts))
        self.assertEqual(2, len(contexts[0].grabs))
        self.assertEqual(
            {
                "left": 110,
                "top": 205,
                "width": 60,
                "height": 30,
            },
            contexts[0].grabs[0],
        )
        self.assertEqual((20, 10), first["seat"].size)
        self.assertEqual((30, 15), second["board"].size)
        self.assertEqual((30, 20, 10), first["seat"].getpixel((0, 0)))

        source.close()
        self.assertTrue(contexts[0].closed)

    def test_invalid_monitor_and_rectangles_fail_before_capture(self):
        with self.assertRaises(ValueError):
            MSSRegionFrameSource(
                0,
                {"table": {"left": 0, "top": 0, "width": 1, "height": 1}},
            )
        with self.assertRaises(ValueError):
            MSSRegionFrameSource(
                1,
                {"table": {"left": 0, "top": 0, "width": 0, "height": 1}},
            )

    def test_simultaneous_first_capture_cannot_create_two_contexts(self):
        contexts = []
        factory_entered = threading.Event()
        release_factory = threading.Event()
        outcomes = []

        def factory():
            factory_entered.set()
            release_factory.wait(timeout=2)
            context = FakeMSS()
            contexts.append(context)
            return context

        source = MSSRegionFrameSource(
            1,
            {
                "table": {
                    "left": 0,
                    "top": 0,
                    "width": 10,
                    "height": 10,
                }
            },
            mss_factory=factory,
        )

        def capture():
            try:
                source.capture()
                outcomes.append("captured")
            except RuntimeError as error:
                outcomes.append(str(error))

        first = threading.Thread(target=capture)
        first.start()
        self.assertTrue(factory_entered.wait(timeout=1))
        second = threading.Thread(target=capture)
        second.start()
        release_factory.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertEqual(1, len(contexts))
        self.assertEqual(1, outcomes.count("captured"))
        self.assertEqual(1, sum("cross capture threads" in item for item in outcomes))


if __name__ == "__main__":
    unittest.main()
