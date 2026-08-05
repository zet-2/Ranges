# Low-cost server validation

## Short answer

The Mac is not needed while the solver is being validated. It collects
completed, owned-simulator observations locally and archives both the raw
snapshot and the canonical public-hand prefix. The solve server then replays
fixed corpora offline. No Gemini or Anthropic request is made by the
solve-only path, so the recurring cost is server runtime and storage only.

This separation makes infrastructure testing inexpensive. It does not make a
commercial solver license free, and it does not by itself prove that a
six-player strategy is near equilibrium.

## Validation pipeline

### 1. Free contract gate

Run the Python and Rust suites before renting a large machine:

```sh
.venv/bin/python -m unittest \
  tests.test_gto_hand_history \
  tests.test_gto_event_collector \
  tests.test_gto_capabilities \
  tests.test_gto_external_backend \
  tests.test_gto_remote \
  tests.test_gto_remote_client \
  tests.test_live_gto \
  tests.test_live_blueprint

cargo test --manifest-path gto_oracle_engine/Cargo.toml --locked
```

This gate checks strict JSON, action order, blinds/antes, minimum raises,
short all-ins, chip conservation, board order, request authentication and
identity, timeouts, byte limits, idempotency, cache behavior, capability
claims, and failure isolation. It tests correctness of the boundary, not the
quality of an equilibrium.

### 2. Solve-only smoke gate

Build the engine once on the server, then solve the built-in transparent
corpus without any model calls:

```sh
python3 gto_oracle_benchmark.py solve \
  --offline-confirmed \
  --engine /opt/gto-oracle/bin/gto-oracle-engine \
  --oracle-cache /var/lib/gto-oracle/validation.sqlite3 \
  --report /var/lib/gto-oracle/oracle-validation.json
```

The report records the exact case key, solver/version, target and achieved
exploitability metadata, iterations, elapsed time, convergence flag, and
whether the result was fresh or cached. A second identical run should be
cache-only. This command validates only the bundled heads-up postflop oracle;
its report says so explicitly.

### 3. Representative corpus

Create a versioned, stratified case set rather than solving random hands
without coverage accounting. At minimum, vary:

- preflop path and number of live players;
- position, starting-stack family, rake and ante;
- flop texture, turn and river runout;
- pot and SPR buckets;
- check, call, donk, bet, raise, short all-in and reopened-action paths;
- every configured sizing menu and its boundary values;
- folded-range/bunching families;
- pure, mixed and low-frequency actions.

Keep a small smoke partition, a medium pull-request partition, and a large
nightly or on-demand partition. Each case must have a stable ID and a complete
input fingerprint. Never silently replace an unsupported size or missing
range.

### 4. Mathematical quality gate

For every fresh solve, require structured—not prose-only—metadata:

- convergence reached under the declared metric;
- exploitability or multi-player regret below the versioned threshold;
- all action frequencies finite, non-negative and summing to one;
- all returned actions legal at the exact node;
- counterfactual EVs finite and in consistent units;
- reach mass conserved through every action and chance transition;
- identical input and solver version reproduce the same policy within a
  declared numeric tolerance.

Repeat a stratified subset at increasing iteration budgets. The measured error
should improve or remain within a justified numerical tolerance. A backend
that returns only a recommended action cannot pass this quality gate; its
offline adapter must expose full policies, EVs and convergence diagnostics to
the validation harness even though the live HTTP response stays minimal.

For multi-player CFR-style solvers, do not reuse a heads-up exploitability
label unless the backend actually computes it. Declare the precise regret or
NashConv-like metric, averaging convention, units and stopping rule.

### 5. Independent cross-check

Use a second trusted solver on a small, fixed sample. Compare complete action
frequencies and EVs at the same ranges, stack, rake, tree and precision—not
only the selected action. Record both solver versions and every abstraction.
Disagreement above tolerance blocks release and becomes a regression fixture.

This is the only phase that may require a separate commercial license. Sampling
keeps it small: there is no reason to pay a reference solver for every cached
case.

### 6. Robustness and load gate

Replay valid and deliberately invalid requests through the server:

- cold and warm cache;
- duplicate idempotency keys;
- concurrent requests while the one solve slot is occupied;
- adapter crash, hang, malformed response and oversized response;
- process restart between streets;
- memory ceiling and out-of-disk behavior;
- stale or tampered hand transcript;
- TLS/authentication failure;
- latency percentiles by tree family.

The required result is either a bound, legal, converged decision or an explicit
fail-closed status. A timeout must never become a guessed action.

## Keeping compute spend low

- Run contract tests and tiny river/turn cases before allocating a large node.
- Use one solver process at a time and enforce hard time, RAM and disk limits.
- Persist the content-addressed solve cache on inexpensive storage.
- Reuse the same solved tree for every private combo and descendant node it
  covers.
- Stop the instance automatically when the batch queue is empty.
- Run the large corpus only after code, manifest or solver-version changes.
- Cross-check only a stratified sample with a paid reference solver.
- Store reports and fingerprints; do not recompute unchanged cases.

A rented server can therefore be used only for the hours of fresh solving.
After the cache and reports are produced it can be shut down. Exact cost
depends on the selected backend's CPU/RAM needs; a full multiway preflop-to-
river solver can require substantially more memory and time than the bundled
heads-up engine.

## Release criteria

The backend may be labelled ready only when all of these are true:

1. Its capability manifest is truthful and has no derived six-max gaps.
2. Every contract, corpus, convergence, cross-check and robustness gate passes.
3. The precise game abstraction and off-tree policy are versioned.
4. The Mac supplies a gap-free public transcript bound to the decision.
5. The server returns only legal actions and fails closed on all uncertainty.

Until then, the server can still be tested extensively, but the product must
report the exact narrower scope rather than the phrase “complete GTO”.
