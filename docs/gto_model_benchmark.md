# Offline Claude versus GTO-oracle benchmark

`gto_oracle_benchmark.py` compares model-selected actions with the local exact
oracle after a session. It is a separate subsystem: it does not import
`poker_assistant.py`, capture the screen, use Gemini/vision, listen for hotkeys,
or emit live advice.

Every benchmark run requires `--offline-confirmed`, which means the poker client
is closed and the evaluated hands are already complete. Anthropic calls are
disabled by default even after that confirmation. They require the additional
`--call-models` flag, so a normal cache-only run cannot incur API cost.

## Transparent demo and validation

The built-in validation suite contains three deterministic fixtures:

- the small Brown river value/bluff game used by the Rust engine tests;
- a turn root with five explicit weighted combos per player and one 50%-pot bet
  size;
- a broader river root with 22 explicit combos per player and 33%/75% bet
  sizes, which adds pure, mixed, and sizing decisions.

It is useful for checking the full pipeline, but it is far too small to rank
models generally. Export it to see the complete strict case schema:

```sh
source .venv/bin/activate
python gto_oracle_benchmark.py write-demo \
  --output benchmark_data/gto_oracle/demo.json
python gto_oracle_benchmark.py validate \
  --cases benchmark_data/gto_oracle/demo.json
```

A case file declares `usage: offline_post_session_only` and contains one or
more complete canonical `SolveSpec` objects. No range, rake, sizing, convergence
target, engine commit, or chip-unit default is silently supplied by the loader.

## Run without model calls

This solves or loads the three oracle fixtures and consumes only already-cached
model responses. A cache miss is reported and makes the run incomplete:

```sh
python gto_oracle_benchmark.py run \
  --offline-confirmed \
  --model both
```

To make new paid provider calls, add the separate authorization flag:

```sh
python gto_oracle_benchmark.py run \
  --offline-confirmed \
  --call-models \
  --model both \
  --cases benchmark_data/gto_oracle/demo.json
```

The defaults are `claude-haiku-4-5` and `claude-sonnet-5`. Override them with
`--fast-model` and `--coach-model`. Structured output restricts a provider
response to one exact action index from the supplied action menu. The prompt
contains the board, exact combo, ranges, stack, pot, rake, unit scale, and legal
tree actions, but never contains oracle frequencies or EVs.

Runs are paired: both models receive the same private-combo decisions. With
`--limit`, decisions are sampled without replacement, probability-proportional
to oracle reach weight, and reproducibly controlled by `--seed`.

Responses resume from
`benchmark_results/gto_oracle/model_responses.jsonl`; solver policies use the
transactional SQLite oracle cache. A cached or fresh solve is rejected unless
it actually reached its requested exploitability target. Each invocation emits
a result JSONL and an aggregate report JSON under
`benchmark_results/gto_oracle/`.

## Measured validation run (2026-07-15)

The complete three-fixture run evaluated 29 OOP private-combo decisions per
model. Both models returned a legal action for every decision. The primary
metric below is reach-weighted action EV regret divided by the starting pot;
lower is better.

| Model | Weighted regret / pot | Unweighted regret / pot | Median uncached latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Claude Haiku 4.5 | 1.529% | 3.516% | 1.20 s | 2.20 s |
| Claude Sonnet 5 | 0.529% | 1.182% | 2.71 s | 4.86 s |

Sonnet's weighted regret was 65.4% lower in this run. That aggregate does not
make Sonnet a GTO policy: on the broader river node, Haiku checked 21 of 22
combos while Sonnet selected the 75%-pot bet for all 22. Both therefore
collapsed toward almost constant actions instead of reproducing the solver's
range-dependent mixed strategy. The paired decision count was also even
(12 lower-regret decisions each, with 5 ties); Sonnet's winning errors were
simply smaller in aggregate.

Treat this as directional evidence that Sonnet is the better temporary quality
choice for these fixtures, while Haiku remains faster. It is not a general
model ranking or a GTO certification: the sample contains only three heads-up
postflop OOP roots and does not yet include facing-bet, preflop, six-max, or
multiway decisions.

The machine-readable summary is
[`gto_model_benchmark_2026-07-15.json`](gto_model_benchmark_2026-07-15.json).
It records the raw artifact paths and SHA-256 hashes, solver convergence for
each fixture, exact metrics, and limitations. The source report is
[`20260715T155050302540Z_report.json`](../benchmark_results/gto_oracle/20260715T155050302540Z_report.json).

## Metrics and limits

For every private combo, the benchmark records:

- poker legality and whether the exact size exists in the solved tree;
- oracle probability assigned to the selected action;
- selected-action EV, best available EV, and EV regret;
- regret divided by starting pot for comparison across chip scales;
- near-optimal coverage, latency, tokens, and cache status.

The paired report counts lower-regret wins/ties and the mean Sonnet-minus-Haiku
regret delta on identically scored decisions.

The current benchmark corpus evaluates only an **OOP root check/bet (or all-in) decision**.
It does not represent a facing-bet fold/call node, preflop, six-max, or multiway
GTO. Results are conditional on the supplied ranges and discrete action tree.
This benchmark evaluates models; it does not train them, make them GTO, or
certify safe real-time play.
