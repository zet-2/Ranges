# Local GTO oracle

This component supports two explicit execution contexts: reproducible offline
benchmarks and live decisions inside a simulator controlled by the user. The
wire protocol requires exactly one acknowledgement and records the selected
context in provenance; neither context can impersonate the other.

`live_gto.py` combines a separately validated six-max preflop blueprint with the
owned-simulator context for conservatively reconstructed HU street roots and
first check/bet descendants. Offline model evaluation remains separately gated
by `--offline-confirmed` and never observes capture or keyboard state.

## Engine decision

The initial oracle uses
[`b-inary/postflop-solver`](https://github.com/b-inary/postflop-solver) pinned to
commit `9d1509fe5077d019825f833eed04b16d342dfda1`.

Why this engine:

- it returns the complete mixed strategy, not only one selected action;
- it exposes per-combo and per-action EV estimates, equity, reach weights, and
  exploitability;
- it solves without hand-strength abstraction, apart from exact suit/chance
  isomorphisms;
- its output can therefore be scored by EV regret instead of brittle
  action-label agreement.

Important limitations:

- heads-up postflop only (OOP versus IP);
- the local Rust engine has no preflop or strategically multiway solve;
- the ranges, rake, starting pot, effective stack, and legal sizing tree must be
  supplied explicitly;
- the upstream project is AGPL-3.0-or-later and its open-source development has
  been suspended since 2023.

One screenshot at every Hero decision supplies enough checkpoints for four
auditable paths: the untouched OOP root, IP after OOP checks, IP facing the
first OOP bet, and OOP facing IP's bet after a previously captured Hero check.
The router verifies contribution, pot, stack, position, and legal-action
invariants before issuing `solve_node`. Prior calls/raises or contradictory OCR
remain unsupported. Multiway snapshots must also remain unsupported; silently
dropping opponents would not produce a GTO answer.

## Preflop blueprint and range handoff

Preflop is not computed from scratch by the Rust engine. The live router reads
the public PokerStudy [NL v2 API](https://www.pokerstudy.ai/api) through
`preflop_blueprint.py`. According to PokerStudy's
[simulation profile](https://www.pokerstudy.ai/sims), these artifacts come from
a six-max MonkerSolver tree with 5% rake capped at 0.5 BB, uniform published
stack buckets, and no open limping. This is a fixed source abstraction; the
project does not claim to have solved the complete live game independently.

The data path is deliberately local and fail-closed:

1. `sync` fetches exact manifest/spot/node responses and writes canonical,
   checksummed envelopes atomically.
2. Every cache read validates checksum, game/version, stack, action path, the
   169 hand classes, action totals, combo totals, and source rounding bounds.
3. `preflop_history.py` simulates blinds and every raise/call/fold/all-in token,
   then accepts the screenshot only when exactly one source path matches.
4. Action branches must conserve each hand class's cumulative reach; the chosen
   branch replaces rather than multiplies its previous reach.
5. A supported Hero preflop decision returns the cached mixed strategy without
   invoking Claude or the postflop engine.
6. If exactly two players survive preflop, their cumulative ranges become the
   OOP/IP ranges of the local postflop subgame. Hero's observed combo must
   already have positive source reach; it is never injected into a range.

Prepare the 100 BB cache before the app starts:

```bash
python preflop_blueprint.py sync --stack 100 --max-depth 3 --workers 8
python preflop_blueprint.py validate --stack 100
```

Source depth 1 contains the unopened chain, depth 2 covers nodes facing the
first raise and single-raised terminal paths, depth 3 adds common 3-bet
continuations, and depth 4 contains all 9,270 published 100 BB nodes. Runtime
network access defaults off; a cache miss is an explicit failure.

`exact` mode requires the 5%/0.5 BB source rake and all six starting stacks at
one published bucket. `abstract` mode chooses a nearest bucket only when all
relevant effective stacks are within `PREFLOP_BLUEPRINT_MAX_STACK_ERROR_PCT`.
Contribution tolerance is computed per seat, so a large raise never grants the
same slack to a zero contribution or blind. Every accepted mismatch is emitted
as an approximation caveat.

Screenshot invariants include dealer-to-position order, the full six-seat map,
folded and all-in identity, contribution and stack conservation, call amount,
and Check/Call button coherence. A full-price player marked folded on the first
captured flop is retained as a preflop survivor; this prevents a postflop fold
from manufacturing a false heads-up preflop handoff.

Known boundaries are explicit:

- six occupied source positions are required preflop; short-handed maps are not
  inferred from the six-max artifact;
- strategically multiway postflop remains unsupported;
- folded-card bunching is not passed to the HU engine;
- turn/river ranges currently retain preflop reach but are not conditioned on
  skipped earlier postflop actions;
- postflop bet trees are intentionally small and an unseen flop defaults to
  cache-only;
- neither exact source matching nor a bounded abstraction proves a full-game
  six-max Nash equilibrium.

## Measured latency on this Mac

These are local measurements on arm64 macOS 26.3 with the native Rust release
build. Compilation time is excluded. Both players used medium, explicit ranges;
the board was `Td 9d 6h` with `Qc`/`2s` on later streets, the starting pot was
100 units, the effective stack was 400 units (SPR 4), and the stopping target
was exploitability at or below 0.5% of the starting pot.

| Tree profile | River | Turn | Flop | Flop memory |
|---|---:|---:|---:|---:|
| One bet size (`50%`) | 0.002 s | 0.046 s | 3.240 s | 381 MiB |
| Two bet sizes (`33%, 75%`) | 0.009 s | 0.306 s | 51.754 s | 2,687 MiB |

With the two-size profile and a five-times tighter 0.1% target, the same solves
took 0.019 seconds on the river, 0.686 seconds on the turn, and 145.289 seconds
on the flop (360 iterations). The flop tree still used about 2,687 MiB.

The timings demonstrate why there is no single "solver response time". Adding
one legal size changed this particular flop solve from about 3.2 seconds to
about 52 seconds and increased estimated tree memory roughly sevenfold. Wider
ranges, deeper stacks, more raise sizes, rake/bunching options, or a tighter
exploitability target can increase the cost further.

An identical cached solve can be returned immediately. Every fresh solve,
regardless of context, records its target, tree, ranges, elapsed time, memory,
iterations, execution context, solver commit, and binary hash.

A full-range owned-simulator turn smoke test of the four live paths measured
about 0.19–0.20 seconds fresh and about 0.03 seconds from SQLite cache on this
machine. These timings exclude the preceding vision reconstruction.

## Evaluation contract

For every private combo at a supported node, store all legal actions with:

- solver frequency;
- per-action EV estimate in engine units;
- combo reach weight and equity;
- node exploitability and convergence settings.

The upstream engine stores and computes EV/action-EV values as `f32`. They are
solver estimates in the integer unit named by the request, not numerically exact
chip values. Upstream applies a shared `starting_pot / 2` EV baseline. The
subtraction in action regret removes that additive baseline, so the regret
metric remains the meaningful comparison even though absolute EV output should
not be presented as exact.

The headline model metric is EV regret:

```text
regret_bb = (best_legal_action_ev - chosen_action_ev) / units_per_bb
```

Also report oracle coverage, exact-size coverage, probability mass assigned to
the chosen action, and near-optimal rates at explicit EV tolerances. A mixed
action is not wrong merely because it is not the highest-frequency action. An
unseen bet size is `OUT_OF_TREE`; it must not be snapped silently to the nearest
size.

## Data policy

PokerBench remains a sealed label-agreement regression test. It cannot be
converted into an EV oracle because it omits weighted ranges, the complete
sizing tree, rake, action EVs, and mixed frequencies. Public GTO Wizard scrapes
and derivative datasets are not used for training or calibration because of
provenance and current usage-right restrictions.

The clean dataset path is to generate first-party, versioned labels from the
pinned open solver, split by complete board/range/tree families, and keep the
held-out test solves out of prompts and tuning.

## Independent river cross-check

One exact toy river game was also solved with the independent MIT-licensed
reference implementation from
[`noambrown/poker_solver`](https://github.com/noambrown/poker_solver). The game
used board `2s 3h 4d 6c 7c`, pot 20, stack 10, OOP range `{AA, QQ}`, IP range
`{KK}`, one 10-chip all-in bet, no raises, and no rake.

After one million iterations, the reachable strategy frequencies agreed as
follows:

| Policy | Brown reference | Pinned oracle | Absolute delta |
|---|---:|---:|---:|
| OOP `QQ` bet | 0.3333332554 | 0.3333317040 | 0.0000015514 |
| OOP `AA` bet | 1.000000000 | 1.000000000 | 0 |
| IP `KK` call versus bet | 0.6666669318 | 0.6666666270 | 0.0000003048 |

The maximum probability delta on reach-supported decisions was 0.0000015514.
Centered OOP root EV differed by about 0.000000333 chip and the maximum
per-root-action EV delta was 0.000009861 chip. The Brown reference reported
exploitability near 0.000001 chip and the pinned oracle 0.000008583 chip. A
strategy difference at an unreachable response node was ignored: Nash
strategies need not be unique at nodes with zero reach. The reachable-node
agreement is strong enough for this fixture to serve as a golden regression
test, while broader spot validation remains necessary.

The upstream solver's default build is no longer reproducible unchanged with
current dependencies/Rust: its broad bincode requirement resolves to an
incompatible API, and Rust 1.97 rejects three legacy implicit raw-pointer
autoreferences. The isolated bridge disables the unused serialization feature,
enables Rayon, and scopes the required lint compatibility flag. The upstream
core still passed 40 applicable tests (four ignored) on this machine. Because
the engine is suspended and AGPL-licensed, it should remain an isolated local
oracle unless the distribution/network-source obligations are deliberately
accepted. Brown's MIT implementation remains the independent river gold oracle.
