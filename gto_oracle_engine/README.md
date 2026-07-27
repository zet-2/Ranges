# GTO Oracle Engine

This directory contains a standalone JSON-to-JSON bridge around
[`b-inary/postflop-solver`](https://github.com/b-inary/postflop-solver), pinned to
commit `9d1509fe5077d019825f833eed04b16d342dfda1`.

It runs as a separate local process and supports offline research plus an
explicitly acknowledged user-controlled simulator context. It has no screen,
keyboard, vision, or model integration of its own.

## Scope

- Heads-up No-Limit Hold'em postflop only (OOP versus IP).
- Initial streets: `FLOP`, `TURN`, or `RIVER`.
- Exact card-state enumeration with suit isomorphism; no hand-strength buckets.
- Discounted CFR with a requested maximum iteration count and exploitability
  target.
- Root, verified same-street descendant, or exact flop-to-current continuation
  strategy. Responses include per-combo action frequencies, per-action EV
  estimates, equilibrium EV estimates, equity, weights, convergence, timing,
  and estimated memory.
- `solve_path` traverses exact turn/river chance cards and returns both players'
  action-conditioned conditional ranges at the final node.
- Integer chip units for pot, stack, and action amounts.

This is not a six-max or multiway solver. The caller must supply defensible OOP
and IP ranges; an equilibrium for incorrect ranges or an underspecified action
tree is not ground truth for the original hand.

## Build

The verified toolchain is pinned to Rust 1.97.0 in `rust-toolchain.toml`. The
pinned upstream source also needs this crate's narrowly scoped Rust lint
compatibility flag; it is committed in `.cargo/config.toml`. Keep the build
target outside the repository to avoid filling the workspace.

With a normal rustup-managed `cargo` on `PATH`:

```sh
cd gto_oracle_engine
CARGO_TARGET_DIR="${TMPDIR:-/tmp}/gto-oracle-engine-target" \
  cargo build --release --locked
CARGO_TARGET_DIR="${TMPDIR:-/tmp}/gto-oracle-engine-target" \
  cargo test --release --locked
```

The binary is then at
`${TMPDIR:-/tmp}/gto-oracle-engine-target/release/gto-oracle-engine`.

The verification session used an isolated rustup installation and target:

```sh
cd gto_oracle_engine
CARGO_HOME=/private/tmp/cargo \
RUSTUP_HOME=/private/tmp/rustup \
RUSTUP_TOOLCHAIN=stable-aarch64-apple-darwin \
CARGO_TARGET_DIR=/private/tmp/oracle-engine-target \
  /private/tmp/cargo/bin/cargo build --release --locked
CARGO_HOME=/private/tmp/cargo \
RUSTUP_HOME=/private/tmp/rustup \
RUSTUP_TOOLCHAIN=stable-aarch64-apple-darwin \
CARGO_TARGET_DIR=/private/tmp/oracle-engine-target \
  /private/tmp/cargo/bin/cargo test --release --locked
```

That session's `stable-aarch64-apple-darwin` toolchain was Rust 1.97.0; the
override avoids a network channel refresh while reproducing the verified local
build. Normal installations should rely on `rust-toolchain.toml` instead.

`CARGO_TARGET_DIR` is deliberately supplied by the caller rather than committed
as a machine-specific path.

## Protocol

The executable reads exactly one JSON request from standard input, writes exactly
one JSON response to standard output, and exits. Unknown request fields are
rejected. Schema version 1 supports the legacy `solve_root`, strict same-street
`solve_node`, and cross-street `solve_path` operations. Every request must
select exactly one context:

- offline: `offline_only_acknowledged: true`, with no simulator field;
- owned simulator: `offline_only_acknowledged: false` and
  `owned_simulator_acknowledged: true`.

Neither acknowledgement and both acknowledgements are rejected. The offline
wire shape remains backward compatible.

```json
{
  "schema_version": 1,
  "id": "river-smoke-1",
  "operation": "solve_root",
  "offline_only_acknowledged": true,
  "street": "RIVER",
  "board": ["Td", "9d", "6h", "Qc", "2s"],
  "oop_range": "AdKd",
  "ip_range": "AsKs",
  "starting_pot": 100,
  "effective_stack": 100,
  "chip_scale": 100,
  "chip_unit": "centi-BB",
  "allocation_mode": "uncompressed_f32",
  "bet_sizes": {
    "flop": {
      "oop": {"bet": "50%", "raise": "2.5x"},
      "ip": {"bet": "50%", "raise": "2.5x"}
    },
    "turn": {
      "oop": {"bet": "50%", "raise": "2.5x"},
      "ip": {"bet": "50%", "raise": "2.5x"}
    },
    "river": {
      "oop": {"bet": "50%", "raise": "2.5x"},
      "ip": {"bet": "50%", "raise": "2.5x"}
    }
  },
  "rake": {"rate_pct": 0.0, "cap": 0.0},
  "tree_options": {
    "add_allin_threshold": 1.5,
    "force_allin_threshold": 0.15,
    "merging_threshold": 0.1,
    "turn_donk_sizes": null,
    "river_donk_sizes": "25%"
  },
  "target_exploitability_pct": 0.5,
  "max_iterations": 1000
}
```

`solve_node` uses the same mathematical tree fields plus four mandatory
assertions. Actions always include an explicit nullable `amount`:

```json
{
  "operation": "solve_node",
  "action_history": [{"kind": "CHECK", "amount": null}],
  "expected_current_player": "IP",
  "expected_facing_bet": 0,
  "expected_node_actions": [
    {"kind": "CHECK", "amount": null},
    {"kind": "BET", "amount": 50}
  ]
}
```

The supported descendant paths are `[CHECK]`, `[BET|ALL_IN]`, and
`[CHECK, BET|ALL_IN]`. The engine solves the complete street tree first, then
traverses only exact actions already present in that tree. A missing size,
terminal/chance transition, unexpected actor, facing amount, or node action set
is an error rather than a fuzzy match.

`solve_path` always begins at a three-card `FLOP` root. Its tagged
`path_history` interleaves exact public actions with exact chance cards:

```json
{
  "operation": "solve_path",
  "street": "FLOP",
  "board": ["2c", "7d", "Jh"],
  "path_history": [
    {"type": "action", "action": {"kind": "CHECK", "amount": null}},
    {"type": "action", "action": {"kind": "BET", "amount": 300}},
    {"type": "action", "action": {"kind": "CALL", "amount": null}},
    {"type": "deal", "card": "4s"}
  ],
  "expected_board": ["2c", "7d", "Jh", "4s"],
  "expected_total_invested": [300, 300],
  "expected_current_player": "OOP",
  "expected_facing_bet": 0,
  "expected_node_actions": [
    {"kind": "CHECK", "amount": null},
    {"kind": "BET", "amount": 575}
  ]
}
```

The omitted fields are the same complete ranges, tree, rake, stack, convergence,
and execution-context fields shown in the root request. Before constructing the
game, every observed legal bet/raise/all-in target is added alongside the base
size menu. The solver is never given future board cards at the flop root; each
card is introduced only by its chance step. The engine then verifies final
board, actor, cumulative postflop investments, facing amount, and exact action
set.

`bet_sizes` and `rake` may be omitted. Their effective defaults are a single
`50%` bet, a `2.5x` raise, and zero rake. `tree_options` is required so the
all-in insertion, forced-all-in, action-merging, and donk-tree choices are never
implicit. Donk configuration has three distinct states: `null` inherits that
street's ordinary OOP bet sizes, `""` supplies an explicit empty donk-size list,
and a non-empty Pio-style string supplies custom donk sizes. With an empty list,
the global `add_allin_threshold` may still insert an all-in. Bet-size syntax is
the upstream Pio-style syntax; comma-separated sizes such as `"33%, 75%, a"` are
accepted.

`chip_scale` is the positive integer number of engine units in the caller's
reference unit, and `chip_unit` is its audit label. The solver operates directly
on the integer pot, stack, and action values; these fields pin their meaning
without rescaling them. `allocation_mode` is required and is either
`"uncompressed_f32"` or `"compressed_i16"`. The memory guard uses the selected
mode's estimate, and provenance echoes the effective choice.

Example invocation (replace the path if you chose another target directory):

```sh
/private/tmp/oracle-engine-target/release/gto-oracle-engine < request.json
```

The `root_actions` array defines the order used by every OOP combo's
`root_action_frequencies` and `root_action_evs_units` arrays. IP does not act at
the root, so those two fields are `null` for IP combos. Equity and frequencies
are fractions in `[0, 1]`. EV and action-EV values are `f32` solver estimates
reported in the request's engine units; they are not numerically exact. Upstream
uses `starting_pot / 2` as its EV baseline. Comparing two actions for the same
combo cancels that shared additive baseline, so EV regret is unaffected.
Exploitability uses the same engine unit. `target_exploitability_pct` is a
percentage of the starting pot: `0.5` means 0.5%, not 50%.

For each combo, `range_weight` is the Pio input weight,
`normalized_weight` is the upstream compatibility mass, and `reach_weight` is
that mass normalized across the player's reachable combos to sum to one.
For `solve_node` and `solve_path`, `policies` contains the current actor's
positive-reach combos.
It separately reports `input_range_weight`, equilibrium-conditioned
`path_weight`, raw `joint_compatible_weight`, and
`conditional_reach_weight`. A Hero combo omitted from this set is off the
equilibrium path and must not be assigned an invented policy.
`solve_path.conditional_ranges` applies the same four-weight decomposition to
both OOP and IP, normalized independently at the final public node.

Every successful response repeats `schema_version`, `id`, and the complete
effective configuration under `provenance.effective_request`. Provenance also
records the solver commit, algorithm, abstraction, allocation mode, memory
guard, and execution context. Consumers should reject responses whose
request ID, schema version, commit, or effective configuration differs from the
submitted request.

The engine refuses a tree whose selected allocation estimate is larger than 8
GiB by default. On a larger-memory solve machine, raise the audited guard
explicitly:

```sh
GTO_ENGINE_MAX_MEMORY_GIB=128 \
  /private/tmp/oracle-engine-target/release/gto-oracle-engine < request.json
```

The value must be a whole number from 1 through 4096. It is an allocation guard,
not a request to reserve memory, and its effective byte value is returned in
both solver provenance and memory metadata. Large flop trees can still take
substantial time and memory; start with river fixtures and narrow ranges.

## Test

```sh
cd gto_oracle_engine
CARGO_TARGET_DIR="${TMPDIR:-/tmp}/gto-oracle-engine-target" \
  cargo test --release --locked
```

For a one-shot protocol smoke test after building:

```sh
cd gto_oracle_engine
/private/tmp/oracle-engine-target/release/gto-oracle-engine \
  < examples/river_smoke_request.json
```

The suite includes a deterministic one-street Brown value/bluff game on
`2s3h4d6c7c`: with a 20-unit pot and 10-unit stack, OOP holds AA or QQ against
IP's KK. It checks that AA bets purely, QQ bluffs one third within an explicit
1% tolerance, and exploitability reaches at most 0.001% of the pot. A separate
tree test pins the three null/empty/custom donk semantics. A real cross-street
test solves a flop tree, traverses check/bet/call and an exact turn card, then
verifies that a private combo containing that turn card disappears from the
conditional range.

## License

This bridge links to an AGPL-3.0-or-later dependency and is therefore licensed
under AGPL-3.0-or-later. The complete terms are in [LICENSE](LICENSE); read
[NOTICE.md](NOTICE.md) before distribution or network deployment. This license
statement applies to the isolated Rust bridge and its linked solver dependency;
it does not by itself relicense separate, unlinked repository components.
