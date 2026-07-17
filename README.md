# Ranges

Ranges is a real-time poker table assistant for macOS. It captures calibrated
screen regions, uses Gemini Vision to reconstruct the visible game state, and
optionally asks Claude for a strategy recommendation based on the
current hand history and locally calculated metrics.

## How it works

1. Captures six seat regions, the board, and the action buttons.
2. Combines the crops into a labelled table mosaic plus a focused
   Hero/board/buttons detail image, replacing the original eight image parts.
3. Parses Gemini's compact structured JSON into a game snapshot.
4. Confirms card backs and real action buttons locally, while preserving
   irreversible folds, usernames, and seat positions throughout the hand.
   For calibrated opponent crops, visible card backs are authoritative: an
   occupied opponent without cards is folded, regardless of a conflicting VLM
   status flag. Hero's username is pinned by `HERO_USERNAME`.
5. Calculates hand rank, pot odds, effective stack, and SPR locally.
6. In FAST mode, Gemini first reconstructs the table, then Claude Haiku receives
   that verified state as compact text. Claude does not perform a second OCR pass
   over the cards, avoiding cross-model card and board disagreements.
7. COACH uses the same stable sequential flow with a deeper Sonnet prompt and
   accumulated same-hand context.
8. Recaptures the table after Vision and again after strategy analysis; if it changed, the
   obsolete recommendation is discarded instead of displayed.

Each validated analysis adds a snapshot to the current hand. A confirmed new
hand resets the accumulated history automatically; `n` remains available for a
manual reset.

## Setup

Requires Python 3.10+ and macOS screen-recording/input-monitoring permissions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`, then calibrate the screen regions if necessary:

- `GEMINI_API_KEY` is required for table reconstruction.
- `GEMINI_MODEL` optionally selects the vision model and defaults to
  `gemini-3.1-flash-lite`.
- `GEMINI_MEDIA_RESOLUTION` controls mosaic processing and defaults to `high`.
  HERO and BOARD are also sent as a separate enlarged lossless PNG so small
  heart/diamond and spade/club glyphs are not damaged by JPEG compression.
- `VISION_LAYOUT` defaults to the faster `mosaic`; set it to `legacy` to send
  the original eight separate images for comparison.
- `SAVE_DEBUG_IMAGES` defaults to `0`, avoiding eight synchronous PNG writes
  on every decision. With this setting, the images in `debug_images/` are not
  expected to be the latest normal capture. Set it to `1` while diagnosing.
  Rejected stale analyses are always stored in timestamped folders under
  `debug_images/stale/`; `latest.txt` points to the newest one.
- `ANTHROPIC_API_KEY` is required for strategy advice.
- `HERO_USERNAME` defaults to `biba287` and prevents a neighboring OCR name
  from being assigned to Hero.
- `CLAUDE_FAST_MODEL` is used by the default FAST mode and defaults to
  `claude-haiku-4-5`.
- `PROMPT_MODE` selects the startup fallback mode: `FAST` uses Haiku and
  `COACH` uses Sonnet. The `m` key still toggles it while the app is running.
- `GEMINI_TIMEOUT_MS`, `CLAUDE_FAST_TIMEOUT_SECONDS`, and
  `FAST_REQUEST_TIMEOUT_SECONDS` bound slow FAST requests; their defaults are
  `10000`, `6.5`, and `12.0`. Gemini enforces a 10-second minimum deadline;
  this is only a failure ceiling and does not delay successful responses.
  `CLAUDE_COACH_TIMEOUT_SECONDS` defaults to `15.0`.
- `CLAUDE_MODEL` is used by COACH mode and defaults to `claude-sonnet-5`.

```bash
python calibrate.py
python poker_assistant.py
```

You can pass a monitor number to the assistant, for example
`python poker_assistant.py 2`.

## Controls

- `j`: capture the table and request strategy advice
- `l`: capture and inspect the reconstructed state only
- `n`: start a new hand and reset history
- `p`: show the preflop chart
- `m`: switch between FAST (Haiku) and COACH (Sonnet) mode
- `q` or `Esc`: quit

Analysis runs in a background worker. Repeated `j`/`l` presses while one request
is active are ignored so an old queued request cannot start on the next decision.

Runtime captures, hand histories, debug output, player statistics, and `.env`
are intentionally excluded from version control.

## Offline PokerBench benchmark

`pokerbench_benchmark.py` compares the configured Haiku and Sonnet models with
the 11,000 held-out solver-labelled decisions published by PokerBench. Run this
only as an offline, post-session evaluation with the poker client closed. This
phase-one test sends neutral structured scenarios directly to Claude; it does
not capture the screen, call Gemini, train the model, or exercise the app's live
prompt and deterministic guards. Decisions use Anthropic's constrained JSON
output; Sonnet 5 sampling parameters are omitted and adaptive thinking is
disabled for this direct classification benchmark.

```bash
python pokerbench_benchmark.py download
python pokerbench_benchmark.py validate --no-download

# Small, proportionally stratified samples (100 cases by default)
python pokerbench_benchmark.py run --model fast --limit 100 --no-download
python pokerbench_benchmark.py run --model coach --limit 100 --no-download

# Same deterministic sample for a direct Haiku/Sonnet comparison
python pokerbench_benchmark.py run --model both --limit 1000 --no-download
```

With `--model both`, the limit applies to each model: `--limit 100` therefore
makes 100 Haiku requests and 100 Sonnet requests.

Use `--all` instead of `--limit` to run every loaded case. Note that
`--model both --all` makes 22,000 requests. Calls are cached in
`benchmark_results/pokerbench/cache.jsonl`, so interrupted runs resume without
paying for completed requests. Per-case JSONL output and aggregate reports are
written beside the cache. Five consecutive provider failures stop further API
calls, and an incomplete run exits non-zero instead of looking like a valid 0%
score. The source files and generated reports are ignored by Git.

The headline metrics distinguish action-family agreement (for example, both
choose `BET`) from exact-decision agreement (same action and solver size). This
is a reproducible regression benchmark, not proof that a model is GTO:
PokerBench supplies one selected solver action per case, not mixed frequencies,
per-action EV, regret, or exploitability. A later solver-oracle phase is needed
to measure those quantities. Because the benchmark is public, possible training
data contamination is another reason not to treat its score as certification.

## Solver oracle and owned-simulator live routing

`gto_oracle/` is the solver-neutral foundation for the next evaluation phase:
immutable heads-up postflop specifications, per-combo mixed policies,
counterfactual-EV scoring, deterministic cache keys, and a transactional SQLite
cache. `gto_oracle_engine/` is the isolated Rust bridge to the pinned
open-source solver.

The benchmark path remains offline-only. The engine also accepts a separate,
truthful `owned_simulator` execution context used by `live_gto.py`; it cannot be
enabled accidentally through the offline acknowledgement. Both paths reject
preflop and multiway states instead of presenting a heads-up approximation as
GTO. Setup, measured Apple Silicon latency, licensing, and methodological limits
are documented in [docs/gto_oracle.md](docs/gto_oracle.md). The measured
Haiku/Sonnet comparison and its limitations are documented in
[docs/gto_model_benchmark.md](docs/gto_model_benchmark.md), with a
[machine-readable result summary](docs/gto_model_benchmark_2026-07-15.json).

Build and test the pinned engine with the verified Rust 1.97.0 toolchain:

```bash
cd gto_oracle_engine
CARGO_TARGET_DIR="${TMPDIR:-/tmp}/gto-oracle-engine-target" \
  cargo build --release --locked
CARGO_TARGET_DIR="${TMPDIR:-/tmp}/gto-oracle-engine-target" \
  cargo test --release --locked
cd ..
```

The model benchmark requires an explicit offline confirmation. Its built-in
demo is only a plumbing smoke test; provide a versioned case file for a real
comparison. `--call-models` separately authorizes paid Anthropic requests.

```bash
python gto_oracle_benchmark.py write-demo
python gto_oracle_benchmark.py validate \
  --cases benchmark_data/gto_oracle/demo.json

# Explicit paid Haiku/Sonnet demo, with the poker client closed
python gto_oracle_benchmark.py run --offline-confirmed --call-models \
  --model both --limit 4 \
  --engine "${TMPDIR:-/tmp}/gto-oracle-engine-target/release/gto-oracle-engine"

# Re-run from cached model completions with network calls disabled
python gto_oracle_benchmark.py run --offline-confirmed \
  --model both --limit 4 \
  --engine "${TMPDIR:-/tmp}/gto-oracle-engine-target/release/gto-oracle-engine"
```

A cache-only run exits non-zero when any selected completion is missing; this
prevents an incomplete comparison from looking successful.

### Experimental live GTO in an owned simulator

`STRATEGY_BACKEND=HYBRID` tries the local solver first and uses the selected
Claude mode only when the node is unsupported or the bounded solve fails. The
solver path requires both an explicit feature flag and confirmation that the
target environment is controlled by the user:

```dotenv
STRATEGY_BACKEND=HYBRID
GTO_LIVE_ENABLED=1
GTO_OWNED_SIMULATOR_ACK=1
GTO_ENGINE_PATH=/private/tmp/oracle-engine-target/release/gto-oracle-engine
GTO_RANGE_SOURCE=blueprint
GTO_RAKE_RATE_PCT=5
GTO_RAKE_CAP_BB=0.5

# Strict source match. Use `abstract` only with the explicit bounds below.
PREFLOP_BLUEPRINT_MATCH_MODE=exact
PREFLOP_BLUEPRINT_ALLOW_NETWORK=0
```

Synchronize and validate the public NL v2 artifacts before starting the app.
Depth 2 covers unopened and facing-one-raise nodes; depth 3 also covers common
3-bet continuations. Depth 4 is the complete 100 BB tree (9,270 nodes) and is a
much larger one-time download.

```bash
python preflop_blueprint.py sync --stack 100 --max-depth 3 --workers 8
python preflop_blueprint.py validate --stack 100
```

The cache is checksummed, schema-validated, read with exact keys, and ignored by
Git. With network access disabled, a missing node fails explicitly instead of
being interpolated. The source is PokerStudy's public
[NL v2 API](https://www.pokerstudy.ai/api); its published profile is a
[MonkerSolver six-max tree](https://www.pokerstudy.ai/sims) with 5% rake capped
at 0.5 BB, uniform stack buckets, and no open limping.

At a supported preflop decision the router reconstructs one unique action path,
checks dealer/position order, contributions, all-ins, stack conservation, call
amount, and visible buttons, then returns the exact hand-class mix from the
cached blueprint. Claude and the Rust postflop engine are not called for that
decision. The displayed pure action is a stable per-hand roll from the complete
mix.

When exactly two players reach the flop, the same preflop path supplies their
cumulative reach ranges to the local Rust postflop solver. Current postflop tree
coverage remains narrow: an untouched Hero-OOP root, Hero IP after the OOP
check, either player facing the first bet when the required prior checkpoint is
present, and flop/turn/river only. The inclusive pot and stacks are rolled back
to the verified street root before solving.

`exact` mode requires all six starting stacks to equal a published bucket and
the source rake profile. Practical unequal-stack tables can opt into bounded,
explicit abstraction:

```dotenv
PREFLOP_BLUEPRINT_MATCH_MODE=abstract
PREFLOP_BLUEPRINT_MIN_TOLERANCE_BB=0.05
PREFLOP_BLUEPRINT_SIZE_TOLERANCE_PCT=10
PREFLOP_BLUEPRINT_MAX_STACK_ERROR_PCT=25
```

The router applies sizing tolerance per seat, keeps zero/blind contributions
tight, requires a unique path, and prints every stack/rake/sizing mismatch in
the answer. It refuses a bucket farther than the configured maximum. Abstract
output is an approximation to the fixed source tree, not exact GTO for the live
stacks.

Press `j` before acting at every Hero decision. Each accepted capture is a
checkpoint for the next decision. Multiway postflop, five-/three-handed seat
maps, ambiguous paths, skipped postflop action history, unsupported bet trees,
and missing cache entries are labelled unsupported. `HYBRID` then falls back to
Claude and prints the reason; `STRATEGY_BACKEND=GTO` fails closed.

This is not a complete real-time solution of six-max Hold'em. In particular,
folded-card bunching is omitted from the HU solve, and turn/river ranges are
conditioned on preflop reach but not yet on earlier postflop actions. Those
boundaries are printed in every solver answer. Static `poker_data.json` charts
remain available with `GTO_RANGE_SOURCE=charts`.

Fresh turn and river solves use separate deadlines. Flop defaults to
`GTO_FLOP_CACHE_ONLY=1`, so an unseen flop falls back immediately instead of
starting a potentially large tree. All settings are listed in `.env.example`.

On the current Apple Silicon development machine, an end-to-end router check
of a full-range synthetic turn node took about `0.19s` fresh and `0.03s` from
cache. These are solver-router timings only: the app's total response time also
includes the preceding vision reconstruction, which remains the larger live
latency component.

## Experimental local card reader

`local_card_reader.py` contains an early OpenCV/Tesseract experiment for
reading cards and pot values locally. It is not yet connected to the main
assistant.
