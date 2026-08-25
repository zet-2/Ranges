# Native 100 BB HU delivery plan

Status checked on 2026-08-23.

## Target claim

The intended product scope is a native two-player, 100 BB cash-game profile,
not a blind-vs-blind projection from a six-max hand. Every admitted artifact
must name stack, blinds, rake, chip rounding, action abstraction, card
abstraction, source/version, and verification method.

For the bundled restricted jam/fold reference, Ranges can already make the
precise claim “measured epsilon-equilibrium of a versioned raked HU jam/fold
game,” because it enumerates both players' complete best responses inside that
finite game.

That claim does **not** automatically transfer to a Monker preflop export.
MonkerWare's own [FAQ](https://www.monkerware.com/faq.html) says it does not
calculate full-strategy exploitability for its preflop ranges. A Monker export
must therefore be labelled as a solver-derived, versioned abstract blueprint
unless Ranges also gains an independent full best-response evaluator for the
same combined preflop/postflop abstraction. Iteration count alone is not a
bilateral best-response certificate.

## Implemented groundwork

- The `hu_jam_fold.py` facade plus its `model`, `solver`, and `artifact` modules:
  explicit rake, no-flop-no-drop, unmatched-chip return, deterministic search,
  bilateral full-best-response measurement, strict fingerprints, and
  fail-closed manifests for the restricted reference game.
- `capture_layout.py` and `calibrate.py`: native-HU `S0`/`S4` mapping, hashed
  calibration evidence, explicit human review, monitor-resolution pinning, and
  inactive-slot rejection. No guessed HU coordinates are included.
- Existing runtime: native two-player position semantics and 2–6 seat public
  hand validation already exist; the card-exact Rust postflop engine remains HU.

## Monker feasibility result

Monker is the practical offline preflop candidate, but purchase is still gated.

- The official [product page](https://www.monkerware.com/solver.html) currently
  lists EUR 499, macOS/Windows, a minimum of 8 GB RAM, and states that solvable
  tree size scales with RAM.
- The official [example-tree page](https://monkerware.com/trees.html) estimates
  4.24 GB for its default 100 BB HU NLHE AUTO tree without donk bets. The current
  M4 host has 16 GB, so a first local tree-building spike is plausible; the
  actual raked action tree must still pass Monker's own memory estimator before
  solving.
- The official [guide](https://monkerware.com/guide.html) says large preflop
  trees may require several days and recommends inspecting required memory in
  the settings tab. It documents a GUI workflow, not a supported headless API.
- The official [Viewer page](https://monkerware.com/viewer.html) confirms that
  local ranges can be exported from MonkerSolver. It does not document a stable
  complete-tree schema or convergence-proof export.
- The public [terms](https://monkerware.com/tos.html) make accounts personal and
  warn about prohibited real-time use, but do not explicitly state the rights
  to use locally generated exports inside a private application, allowed
  machine count, or remote-machine activation. Those points require written
  vendor confirmation before purchase/integration.

The existing PokerStudy artifact demonstrates that a Monker-derived tree can
represent a 5% rake capped at 0.5 BB, but it is evidence about that published
artifact, not a substitute for checking our exact rake semantics and Monker
version during the spike.

Before purchase, ask `support@monkerware.com` for written answers to this exact
scope (do not send screenshots, account credentials, or project secrets):

> I want to use MonkerSolver offline to generate a private 100 BB heads-up NLHE
> cash-game preflop solution for analysis in a simulator I own. I will not
> redistribute the solver or generated ranges and will not use it where a poker
> room prohibits real-time assistance. Does one personal license permit (1)
> importing my locally generated range exports into my own private application,
> (2) installation/activation on both my personal Mac and a rented machine used
> only by me for the solve, and (3) retaining those exports after the solve?
> Please also confirm which export contains the complete preflop tree, mixed
> action frequencies, action amounts, ranges/EVs, iteration or convergence
> metadata, and the exact semantics available for percentage rake, cap,
> no-flop-no-drop, chip rounding, and uncalled bets.

## Remaining gates in order

1. Capture a real owned-simulator HU calibration and pass the corpus in
   `hu_capture_calibration.md`.
2. Obtain written Monker confirmation for private generated-export use,
   activation/machine count, and stable preflop export semantics.
3. Build the exact 100 BB HU rake/action profile in Monker and record its native
   tree identity, RAM estimate, abstraction settings, and rake behavior before
   starting a long solve.
4. Produce one golden export and write a strict decoder that checks node reach,
   action frequencies, combo/blocker conservation, stack/action matching, and
   file fingerprints.
5. Decide the honest admission class from available evidence:
   `solver_derived_abstract_blueprint`, or `measured_epsilon_equilibrium` only
   after an independent bilateral best-response evaluator exists for the same
   complete abstraction.
6. Validate preflop-to-postflop range handoff and full native-HU sessions across
   stack drift. Exact 100 BB matching is usable only with auto top-up; otherwise
   additional versioned stack profiles or explicitly disclosed interpolation
   are required.

Until these gates pass, the repository contains strong HU components and a
complete restricted-game certification path, but not a production native-HU
100 BB preflop policy.
