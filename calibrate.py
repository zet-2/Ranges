#!/usr/bin/env python3
"""Create and review versioned PokerStars capture-layout profiles.

Native HU calibration is intentionally a two-step process:

1. ``heads-up`` records explicit crop corners and hashed visual evidence.
2. ``approve`` marks that exact frame as reviewed after a human inspects it.

The live assistant refuses draft profiles and profiles captured at a different
monitor resolution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import mss
from PIL import Image, ImageDraw, ImageFont
from pynput import mouse
from rich.console import Console

from capture_layout import (
    ARTIFACT_TYPE,
    CALIBRATION_METHOD,
    SCHEMA_VERSION,
    CaptureLayoutError,
    load_capture_layout,
    parse_capture_layout,
    sha256_file,
    verified_payload,
)


console = Console()
HU_REGION_ORDER = (
    ("seat1", "opponent pod (canonical S0)"),
    ("hero", "Hero pod and face-up cards (canonical S4)"),
    ("board", "community cards plus literal Pot label"),
    ("buttons", "large Hero action-button area"),
)


def _write_json(path: Path, payload: object, *, force: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise CaptureLayoutError(
            f"refusing to overwrite {path}; choose another path or pass --force"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temporary_path = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary_path, path)
    except OSError as error:
        raise CaptureLayoutError(f"cannot write {path}: {error}") from error
    finally:
        if handle is not None:
            handle.close()
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _click_pair(label: str) -> tuple[tuple[int, int], tuple[int, int]]:
    clicks: list[tuple[int, int]] = []

    def on_click(x, y, button, pressed):
        if not pressed:
            return None
        clicks.append((round(x), round(y)))
        corner = "top-left" if len(clicks) == 1 else "bottom-right"
        console.print(f"[green]✓ {label} {corner}: {clicks[-1]}[/green]")
        return False if len(clicks) == 2 else None

    console.print(
        f"\n[bold cyan]{label}[/bold cyan]: click its top-left, then bottom-right."
    )
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    return clicks[0], clicks[1]


def _relative_region(
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    monitor: dict,
    name: str,
) -> dict[str, int]:
    absolute_left = min(first[0], second[0])
    absolute_top = min(first[1], second[1])
    absolute_right = max(first[0], second[0])
    absolute_bottom = max(first[1], second[1])
    left = absolute_left - monitor["left"]
    top = absolute_top - monitor["top"]
    width = absolute_right - absolute_left
    height = absolute_bottom - absolute_top
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise CaptureLayoutError(f"{name} is not a positive region on the monitor")
    if left + width > monitor["width"] or top + height > monitor["height"]:
        raise CaptureLayoutError(f"{name} extends beyond the selected monitor")
    return {"left": left, "top": top, "width": width, "height": height}


def _save_preview(crops: dict[str, Image.Image], path: Path) -> None:
    canvas = Image.new("RGB", (1000, 700), (18, 20, 23))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    cells = (
        ("OPPONENT S0", "seat1", (8, 8, 496, 346)),
        ("HERO S4", "hero", (504, 8, 992, 346)),
        ("BOARD + POT", "board", (8, 354, 496, 692)),
        ("ACTION BUTTONS", "buttons", (504, 354, 992, 692)),
    )
    for label, name, (left, top, right, bottom) in cells:
        draw.rectangle((left, top, right, bottom), outline=(82, 91, 101), width=2)
        draw.text((left + 8, top + 6), label, fill=(100, 220, 235), font=font)
        image = crops[name].convert("RGB")
        available_width = right - left - 16
        available_height = bottom - top - 40
        scale = min(
            available_width / image.width,
            available_height / image.height,
        )
        resized = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(
            resized,
            (
                left + 8 + (available_width - resized.width) // 2,
                top + 34 + (available_height - resized.height) // 2,
            ),
        )
    canvas.save(path, format="PNG")


def _capture_heads_up(args: argparse.Namespace) -> int:
    output = Path(args.output)
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if output.exists() and not args.force:
        raise CaptureLayoutError(
            f"refusing to overwrite {output}; pass --force only if intentional"
        )
    if evidence_dir.exists() and any(evidence_dir.iterdir()) and not args.force:
        raise CaptureLayoutError(
            f"evidence directory is not empty: {evidence_dir}"
        )

    with mss.MSS() as screen:
        if args.monitor <= 0 or args.monitor >= len(screen.monitors):
            raise CaptureLayoutError(
                f"monitor {args.monitor} is unavailable; choose 1..{len(screen.monitors) - 1}"
            )
        selected = dict(screen.monitors[args.monitor])
        console.print(
            "[bold]Native HU capture calibration[/bold]\n"
            f"Selected monitor {args.monitor}: "
            f"{selected['width']}x{selected['height']}.\n"
            "Keep one owned/simulated HU table stationary and fully visible."
        )
        regions = {
            name: _relative_region(
                *_click_pair(label),
                monitor=selected,
                name=name,
            )
            for name, label in HU_REGION_ORDER
        }

        grabbed = screen.grab(selected)
        frame = Image.frombytes("RGB", grabbed.size, grabbed.bgra, "raw", "BGRX")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    reference_path = evidence_dir / "reference.png"
    frame.save(reference_path, format="PNG")
    crops: dict[str, Image.Image] = {}
    for name, region in regions.items():
        crop = frame.crop(
            (
                region["left"],
                region["top"],
                region["left"] + region["width"],
                region["top"] + region["height"],
            )
        )
        crops[name] = crop
        crop.save(evidence_dir / f"{name}.png", format="PNG")
    _save_preview(crops, evidence_dir / "preview.png")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "layout_id": args.layout_id,
        "table_format": "heads_up",
        "monitor": {
            "index": args.monitor,
            "width_px": selected["width"],
            "height_px": selected["height"],
        },
        "canonical_seats": {"hero": 4, "opponent": 0},
        "regions": regions,
        "calibration": {
            "method": CALIBRATION_METHOD,
            "status": "draft",
            "reference_frame_sha256": sha256_file(reference_path),
            "crop_sha256": {
                name: sha256_file(evidence_dir / f"{name}.png")
                for name in regions
            },
            "reviewed_reference_frame_sha256": None,
        },
    }
    layout = parse_capture_layout(payload)
    _write_json(output, layout.to_payload(), force=args.force)
    console.print(
        f"\n[green]Draft written:[/green] {output.resolve()}\n"
        f"[green]Review this preview:[/green] {(evidence_dir / 'preview.png')}\n"
        "The assistant will refuse this draft. If every crop is exact, run "
        "the approve command printed in the documentation."
    )
    return 0


def _approve(args: argparse.Namespace) -> int:
    layout = load_capture_layout(args.profile, require_verified=False)
    if layout.calibration.status != "draft":
        raise CaptureLayoutError("the supplied profile is not a draft")
    if not args.i_reviewed_crops:
        raise CaptureLayoutError(
            "approval requires --i-reviewed-crops after visually inspecting preview.png"
        )

    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    reference_path = evidence_dir / "reference.png"
    if sha256_file(reference_path) != layout.calibration.reference_frame_sha256:
        raise CaptureLayoutError("reference.png does not match the draft profile")
    with Image.open(reference_path) as source:
        reference = source.convert("RGB")
    if reference.size != (layout.monitor_width_px, layout.monitor_height_px):
        raise CaptureLayoutError(
            "reference.png dimensions do not match the draft monitor"
        )
    for name, expected_digest in layout.calibration.crop_sha256:
        crop_path = evidence_dir / f"{name}.png"
        if sha256_file(crop_path) != expected_digest:
            raise CaptureLayoutError(
                f"{name}.png does not match the draft profile"
            )
        region = dict(layout.regions)[name]
        expected_pixels = reference.crop(
            (
                region.left,
                region.top,
                region.left + region.width,
                region.top + region.height,
            )
        )
        with Image.open(crop_path) as source:
            supplied_pixels = source.convert("RGB")
        if (
            supplied_pixels.size != expected_pixels.size
            or supplied_pixels.tobytes() != expected_pixels.tobytes()
        ):
            raise CaptureLayoutError(
                f"{name}.png is not the declared crop of reference.png"
            )

    approved = verified_payload(layout)
    verified = parse_capture_layout(approved)
    verified.require_verified()
    _write_json(Path(args.output), approved, force=args.force)
    console.print(
        f"[green]Verified layout written:[/green] {Path(args.output).resolve()}\n"
        f"Fingerprint: {verified.fingerprint}"
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    layout = load_capture_layout(
        args.profile,
        require_verified=args.require_verified,
    )
    console.print(
        f"layout_id={layout.layout_id}\n"
        f"table_format={layout.table_format}\n"
        f"status={layout.calibration.status}\n"
        f"monitor={layout.monitor_index}:"
        f"{layout.monitor_width_px}x{layout.monitor_height_px}\n"
        f"active_seats={','.join(map(str, sorted(layout.active_seat_indices)))}\n"
        f"fingerprint={layout.fingerprint}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    heads_up = commands.add_parser(
        "heads-up",
        help="interactively create a draft native-HU layout and hashed evidence",
    )
    heads_up.add_argument("--layout-id", required=True)
    heads_up.add_argument("--monitor", type=int, default=1)
    heads_up.add_argument("--output", required=True)
    heads_up.add_argument("--evidence-dir", required=True)
    heads_up.add_argument("--force", action="store_true")
    heads_up.set_defaults(handler=_capture_heads_up)

    approve = commands.add_parser(
        "approve",
        help="approve one exact draft after human review of its hashed crops",
    )
    approve.add_argument("--profile", required=True)
    approve.add_argument("--evidence-dir", required=True)
    approve.add_argument("--output", required=True)
    approve.add_argument("--i-reviewed-crops", action="store_true")
    approve.add_argument("--force", action="store_true")
    approve.set_defaults(handler=_approve)

    validate = commands.add_parser("validate", help="validate and fingerprint a layout")
    validate.add_argument("--profile", required=True)
    validate.add_argument("--require-verified", action="store_true")
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (CaptureLayoutError, OSError) as error:
        console.print(f"[red]Calibration error:[/red] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
