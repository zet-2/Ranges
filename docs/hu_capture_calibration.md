# Native HU capture calibration

Ranges keeps the vision wire protocol stable: it always uses canonical seats
`S0`–`S5`, with Hero fixed at `S4`. A native two-player table maps the single
opponent pod to `S0`; `S1`, `S2`, `S3`, and `S5` become labelled blank images.
Position still comes from the visible dealer button, so this mapping does not
confuse BTN/SB with BB.

No HU coordinates are committed to the repository. They cannot be inferred
honestly from the existing six-seat crops, and a layout is sensitive to the
PokerStars theme, window geometry, display scaling, and monitor resolution.

## Create a draft from a real HU table

Use only a simulator or table environment you own and are permitted to inspect.
Keep the table stationary and click the requested corners:

```bash
.venv/bin/python calibrate.py heads-up \
  --layout-id pokerstars.hu.owned-sim.v1 \
  --monitor 1 \
  --output debug_images/calibration/hu-v1/draft.json \
  --evidence-dir debug_images/calibration/hu-v1/evidence
```

The command captures one full reference frame and four crops: opponent, Hero,
board/Pot, and action buttons. It also creates `preview.png`. The profile pins
the monitor index/dimensions and SHA-256 of the frame and every crop. Both the images
and profile in the example path are runtime evidence under the git-ignored
`debug_images/` directory.

A draft is structurally valid but cannot be loaded by the assistant.

## Review and approve the exact evidence

Inspect `reference.png`, all four individual crops, and `preview.png`. Each crop
must include its complete semantic target without adjacent amounts or controls.
Then approve that exact hashed frame into a separate file:

```bash
.venv/bin/python calibrate.py approve \
  --profile debug_images/calibration/hu-v1/draft.json \
  --evidence-dir debug_images/calibration/hu-v1/evidence \
  --output debug_images/calibration/hu-v1/verified.json \
  --i-reviewed-crops
```

Approval re-hashes every file and also proves pixel-for-pixel that each stored
crop came from the declared rectangle of `reference.png`. It does not silently
overwrite an existing artifact.

Validate the result independently:

```bash
.venv/bin/python calibrate.py validate \
  --profile debug_images/calibration/hu-v1/verified.json \
  --require-verified
```

Activate it in `.env`:

```dotenv
VISION_LAYOUT=mosaic
POKER_CAPTURE_LAYOUT_PATH=debug_images/calibration/hu-v1/verified.json
```

At runtime, Ranges fails closed when:

- the profile is a draft, malformed, or has unknown fields;
- Hero is not canonical `S4` or the HU opponent is not canonical `S0`;
- a rectangle is outside the calibrated monitor;
- the selected monitor index or resolution differs from the profile;
- Gemini emits player content in an inactive canonical slot;
- the dealer is not `S0` or `S4`.

The embedded six-seat coordinates remain unchanged when
`POKER_CAPTURE_LAYOUT_PATH` is unset.

## What this verification does not prove

`verified` means that one exact set of rectangles was visually reviewed. It is
not a claim that semantic reconstruction is production-ready. Before native HU
capture is admitted for an end-to-end strategy profile, an owned-simulator
corpus still has to cover, at minimum:

- Hero and opponent as dealer across preflop, flop, turn, and river;
- every legal large action-button combination and no-action frames;
- fold, check, call, bet, raise, all-in, and uncalled-chip transitions;
- short stacks, auto top-up behavior, changing pot labels, and suit glyphs;
- changed display scaling/window position and the intended resolution-drift
  rejection;
- full-hand transcript continuity without an invented invisible check.

Until that corpus passes, continuous semantic decoding remains experimental and
the absence of a native-HU preflop blueprint remains a separate strategy gap.
