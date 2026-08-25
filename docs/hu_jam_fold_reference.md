# Raked HU jam/fold reference harness

`hu_jam_fold.py` is an integration and certification harness for one deliberately
restricted heads-up game. It is not a preflop solver for unrestricted HUNL and
must never be presented as one.

The public import/CLI facade remains `hu_jam_fold.py`. The implementation is
split into `hu_jam_fold_model.py` (rake, payoffs, domain types),
`hu_jam_fold_solver.py` (RM+ search and bilateral best responses), and
`hu_jam_fold_artifact.py` (strict JSON and certificate verification). Each
module stays below 800 lines and the split does not alter artifact fingerprints.

## Game and rake contract

The public tree is fixed:

1. BTN/SB chooses `fold` or `jam`.
2. After a jam, BB chooses `fold` or `call`.
3. A call goes directly to showdown.

An input artifact supplies a finite joint distribution over private SB/BB
types and the SB showdown equity for every reachable type pair. Sparse joint
distributions are permitted, so blocker-dependent private models can be
represented without pretending that the two ranges are independent.

The rake profile is part of the game fingerprint. It pins:

- percentage, cap, and chip unit;
- floor-to-chip rounding;
- deduction from the awarded pot after unmatched chips are returned;
- `no_flop_no_drop` behavior.

With `no_flop_no_drop=true`, an SB fold and a BB fold are unraked. A called jam
reaches a runout and pays rake. If BB folds to a jam, the unmatched part of the
SB stack is returned before the awarded pot is calculated.

## What the certificate means

The bundled search procedure is deterministic simultaneous regret-matching+
with linear averaging. Running that algorithm is not treated as proof of
convergence, particularly because rake makes the players' monetary utilities
non-zero-sum.

For every candidate strategy, the verifier instead enumerates the best pure
action independently at every private-type information set for each player.
It reports:

- current SB and BB utility;
- each player's best-response utility;
- SB and BB unilateral deviation gain;
- `epsilon = max(SB gap, BB gap)`;
- expected rake, reconstructed as the negative sum of player utilities.

The enumeration covers the complete supplied finite game. Numeric evaluation
uses IEEE-754 `float64`, declared in the solution manifest. “Full” refers to
best-response coverage of that artifact, not to full Hold'em. Strict loading
recomputes every value, verifies both fingerprints, and rejects an unmet target
or any manifest that claims `full_hunl=true`.

The precise claim is therefore:

> Measured epsilon-equilibrium of a versioned raked HU jam/fold game with an
> explicit finite private-type/payoff model.

## Included integration reference

[`tests/fixtures/hu_jam_fold_two_type_raked_v1.json`](../tests/fixtures/hu_jam_fold_two_type_raked_v1.json)
is a two-private-type fixture used to exercise the complete machinery. Its
5%-capped rake profile includes no-flop-no-drop. Its equity values are explicit
test-game payoffs; they are not a 169-class or 1,326-combo Hold'em equity
artifact and are not usable as a poker strategy.

Solve it and write a diagnostic/verified manifest:

```bash
.venv/bin/python hu_jam_fold.py solve \
  --game tests/fixtures/hu_jam_fold_two_type_raked_v1.json \
  --output /private/tmp/hu-jam-fold-solution.json \
  --target-epsilon-bb 0.0005 \
  --max-iterations 10000 \
  --min-iterations 2000 \
  --check-every 1000
```

Recompute both best responses under an independently supplied maximum:

```bash
.venv/bin/python hu_jam_fold.py verify \
  --game tests/fixtures/hu_jam_fold_two_type_raked_v1.json \
  --solution /private/tmp/hu-jam-fold-solution.json \
  --max-epsilon-bb 0.0005
```

`solve` returns non-zero when its declared target is not reached, although it
still writes the diagnostic manifest. `verify` and the Python `load_solution`
API reject that manifest by default. Existing output is never overwritten
unless `--force` is explicit.

## Gate before a poker artifact is admissible

Replacing the two-type integration fixture requires all of the following:

1. documented provenance and redistribution/use rights for the private model;
2. blocker-correct joint deal weights;
3. rake values matching the target environment exactly;
4. independently reproducible showdown equities or terminal payoffs;
5. a fingerprinted artifact and a solution passing bilateral best-response
   verification at the declared epsilon;
6. an explicit statement of every abstraction boundary.

Until those gates pass, this harness de-risks loading, rake accounting,
certification, labeling, and fail-closed behavior only. It supplies no live
preflop policy.
