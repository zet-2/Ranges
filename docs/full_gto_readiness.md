# Full-GTO readiness

## Decision

The application now defaults to strict solver-only routing. It never calls
Claude from `STRATEGY_BACKEND=GTO`, rejects state abstractions, and fails closed
when the available policy is not an exact match to its declared source game.

This is not yet a full six-max no-limit Hold'em Nash solver. The primary blocker
is solver and state architecture, not the current laptop alone.

The repository now has the two integration boundaries needed to add such a
backend without moving table capture off the Mac:

- a canonical, replayable 2–6 player public-hand transcript with exact
  fold/check/call/bet-to/raise-to/all-in and board events;
- a strict external-process adapter plus an auditable capability manifest that
  cannot silently relabel the bundled HU backend as full six-max.

What remains missing is the actual licensed or newly developed multiway solver
behind that adapter, plus a continuous event source that observes every public
transition.

## Existing-backend audit

Checked against public vendor material on 2026-07-25:

- [`b-inary/postflop-solver`](https://github.com/b-inary/postflop-solver) is the
  pinned open-source dependency already in use. It is a strong card-exact HU
  postflop engine, not a preflop or multiway solver.
- [MonkerSolver](https://www.monkerware.com/solver.html) advertises Hold'em from
  any street with any number of players, but its official workflow is a GUI and
  the public guide does not document a server/CLI API. It costs €499 and uses
  configurable card abstraction for large games.
- [PioSOLVER](https://piosolver.com/products/) provides a documented scripting
  interface, but its product scope is heads-up.
- [NexusGTO](https://www.nexusgto.com/) advertises six-player solving and an
  [API](https://www.nexusgto.com/docs), but the public product still says
  “Postflop coming soon,” API access is private beta, postflop is metered, and
  the computation is vendor-hosted rather than installed on our solve server.
- [ART/GTO](https://artgto.com/) is a local postflop solver with documented
  exports, but its published workflow describes two supplied ranges and does
  not provide the missing six-player preflop-to-river backend.

No audited option is therefore a drop-in, open-source, server-owned,
six-player preflop-to-river engine with the structured convergence output this
project requires. Monker is the closest local multiway solver; Nexus is the
closest documented web API. Either needs explicit vendor access, licensing and
an independently tested adapter before it can replace the truthful native
manifest.

## What is exact enough to return in strict mode

- A cached PokerStudy NL v2 preflop node whose six starting stacks, action path,
  rake, and sizing match the fixed source profile exactly.
- A mixed action realized with a private HMAC secret and memoized
  deterministically by hand and node.

The source remains a rounded, fixed-action MonkerSolver blueprint. Its checksum
and reach conservation are validated locally, but its published payload does
not include a full-game exploitability certificate.

## What intentionally fails closed

- Abstract stack buckets or tolerated preflop sizing mismatches.
- Static chart ranges.
- Any strategically multiway postflop state.
- Any six-max-origin postflop solve while folded-card bunching is omitted.
- Turn or river nodes without a gap-free transcript from the true flop root.
- Ambiguous or skipped public action history.
- An observed aggressive size that is illegal or cannot be added to the
  declared discrete solver tree.
- A solve that misses its requested exploitability target or deadline.

`HYBRID` remains available only as an explicit opt-in. It may display a
conditional solver result under `APPROXIMATE_SOLVER` or fall back to Claude, but
neither path is presented as strict GTO.

## Hardware

The current host is a fanless 10-core Apple M4 MacBook Air with 16 GB RAM. Local
measurements for one representative full-range flop family were:

| Sizes per street | Uncompressed memory | Approximate time |
| --- | ---: | ---: |
| 1 | 0.40 GB | 3.3 s to 0.5% pot |
| 2 | 2.82 GB | 51.8 s to 0.5%; 145.3 s to 0.1% |
| 3 | 7.52 GB | not latency-qualified |
| 4 | 16.45 GB | does not fit safely on this host |

The engine uses Rayon on CPU. A larger GPU does not accelerate it. For serious
offline HU cache generation, use:

- 128 GB RAM, 16–32 high-performance CPU cores, active cooling, and at least
  2 TB NVMe for multi-size work;
- 256–512 GB ECC RAM and 32–64 CPU cores for wider trees, bunching, deep stacks,
  or concurrent batch solving.

`GTO_ENGINE_MAX_MEMORY_GIB` now raises the engine's audited 8 GiB default guard
on such a machine. The limit is reported in every result. Increase it only to a
value below physical memory.

More hardware improves conditional heads-up subgames and offline coverage. It
does not add preflop solving, multiway game logic, missing action history, or a
continuous no-limit action space.

The implemented remote boundary is after local OCR/history validation and
before backend evaluation. Protocol v2 optionally carries the complete public
transcript; a backend declaring full-six-max support requires it and the server
replays it independently. A rented Linux host can therefore own the blueprint,
persistent cache, range reconstruction, and solver process without receiving
screenshots or OCR-provider secrets. The Mac accepts only an authenticated
response whose request ID and full-state fingerprint match, then performs its
existing final freshness recapture. Deployment details are in
[`gto_remote/README.md`](../gto_remote/README.md).

## Architecture required for broader GTO coverage

The native HU path now losslessly reconstructs a solved flop game, adds exact
observed aggressive sizes, traverses every action/chance node, and exports both
conditional ranges through river. Its SQLite cache is keyed by the complete
path. The remaining work below is therefore continuous capture, performance
reuse, bunching, abstraction breadth, and multiway coverage—not basic HU range
continuity.

1. Feed the implemented event recorder from a continuous owned-simulator source.
   The current recorder already validates every public check, fold, call,
   bet-to, raise-to, all-in, board card, stack, pot, blind, ante, and rake
   profile with chip conservation, but manual Hero-decision captures can skip
   opponent transitions and therefore make the transcript unavailable.
2. Replace one-shot lossless reconstruction with an optional persistent
   per-hand solver daemon so later decisions can reuse the already solved flop
   game instead of solving the same root for each new path.
3. Expand the current versioned one-size/2.5x tree into representative
   per-street and per-position sizing menus. Exact observed off-tree sizes are
   already added alongside—not substituted for—the configured base menu.
4. Feed the four folded preflop ranges into the upstream bunching calculation
   and validate its extra memory/time cost.
5. Measure conditional-node regret in addition to root exploitability, with
   independent solver cross-checks over board, range, SPR, rake, and tree
   families.
6. Build or license a compatible preflop continuation blueprint with the same
   postflop game abstraction and convergence metadata.
7. Use a separate multiway solver or blueprint pipeline. The pinned Rust
   dependency is mathematically heads-up-only; no hardware upgrade can change
   that interface. The external backend adapter is ready to host one, but no
   proprietary solver executable or license is bundled.

The achievable engineering claim is a tightly converged epsilon-equilibrium for
an explicitly versioned game abstraction. Literal exact GTO for the continuous,
full six-max no-limit game is not a practical finite-compute target.

The low-cost, server-only acceptance process is specified in
[`gto_server_validation.md`](gto_server_validation.md). It separates free
contract/replay tests from solve convergence, representative corpus coverage,
independent reference checks, and load testing.
