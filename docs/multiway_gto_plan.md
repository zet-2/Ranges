# Multiway GTO implementation and operations plan

## Bottom line

The repository now contains the transcript-first protocol, structured response
contract, capability binding, server/client routing, and an experimental
default-off semantic decoder that turns stable continuous screenshots into
transactionally accepted public-hand observations. It does **not** contain a
Monker-specific adapter or Monker output decoder, and the capture path has not
passed the production acceptance corpus below. It is therefore not yet a
working full-game six-max solver.

The practical target is a measured epsilon-equilibrium for an explicitly
versioned finite game abstraction. Literal exact GTO for continuous six-max
no-limit Hold'em is not a realistic finite-compute claim. In a multiplayer
game, the proof must also name a suitable solution concept and multiplayer
convergence metric; heads-up exploitability cannot be reused without
qualification. This is not merely a product limitation: the standard CFR
epsilon-Nash guarantee applies to two-player zero-sum perfect-recall games,
while [published multiplayer poker work](https://poker.cs.ualberta.ca/publications/AAMAS10.pdf)
explicitly notes that the general multiplayer guarantee is lost.

## Backend feasibility snapshot (2026-07-30)

| Candidate | Publicly documented coverage | Automation fit | Current decision |
| --- | --- | --- | --- |
| [MonkerSolver](https://www.monkerware.com/solver.html) | Hold'em from any street with any number of players; customizable abstract betting trees; EUR 499 license. | The public guide documents a Java GUI and manually built/started trees, not a supported headless API or a complete machine-readable proof export. | Most plausible self-hosted engine, conditional on written automation/export/license confirmation and a golden native export. |
| [GTO Wizard Ultra](https://blog.gtowizard.com/introducing-multiway-preflop-solving/) | Up to nine players preflop and three players postflop in the current public release. | Its [terms](https://gtowizard.com/terms/) prohibit automated requests/scripts and live-game use. | Not an adapter target for this project. |
| [NexusGTO beta](https://www.nexusgto.com/docs) | The public beta page claims a six-player REST API, preflop lookup, postflop solving, and hand-history parsing. | Programmatic API is promising, but access is manual/private-beta and the public example does not establish all v3 proof, convergence, bunching, or full action-history semantics. | Secondary candidate only after a real sandbox response, contract/license review, and independent output validation. |
| Simple 3-Way | Exactly three postflop players. | Desktop product rather than a verified six-player server contract. | Insufficient for the requested 3–6-player path. |

The absence of a supported adapter contract is currently a stronger blocker
than CPU or RAM. No installed licensed multiway solver, native multiway tree,
ready capability manifest, or genuine 3–6-player solver export was found in
the development environment. The bundled Rust engine remains heads-up only.

## Delivery status

| Area | Implemented now | Still required |
| --- | --- | --- |
| V3 request | `MultiwayDecisionState` carries only the canonical public hand, Hero seat/cards, and capture ID. Street, board, pot, stacks, contributions, actor, and legal actions come from replay. Strict keys, Decimal strings, byte limits, duplicate-key rejection, and decision fingerprints are implemented. | Exercise the contract against the real solver adapter and a production corpus. |
| Structured outcome | A solved response must contain a complete mixed policy and proof: action kinds/sizes, frequencies, optional EVs, backend identity, profile/abstraction IDs, solution concept, convergence metric/value/target, iterations, and approximation disclosure. Policy legality and convergence are checked before advice is rendered locally. | Decode real solver artifacts into this schema without inventing missing EV or convergence data. |
| Capability binding | The client pins a local capability manifest; the server loads the external backend manifest; solved proof is bound to backend ID, version, profile/abstraction IDs, solution concept, metric, target, and canonical manifest fingerprint. Per-state support gaps fail closed. Exactness gaps force `approximate=true`; strict `GTO` rejects them while `GTO_MULTIWAY` may display the explicit finite-game approximation. | Install the same audited manifest at both ends and operationally pin the permitted game-profile and abstraction IDs. Declare only capabilities verified against the real backend. |
| HTTP/server route | The authenticated endpoint parses v2/v3 and dispatches only schemas advertised by its router: native is v2-only; external is v2/v3. It preserves request/fingerprint idempotency, limits bodies, and serializes solver work. The generic no-shell external-process boundary handles v3 structured results. | Point the boundary at a licensed solver adapter or a thin shim to a persistent solver daemon. Add daemon-aware readiness. |
| Application route | `GTO_REMOTE_PROTOCOL=multiway-v3` selects the v3 client. The application builds a capture ID and compares hand ID, actor, street, board, seats, pot, stacks, bets, statuses, call amount, Hero controls, and Hero combo with replay before sending. A revocable accepted-history token is checked before transport, after transport, and after the final pixel guard. `GTO_MULTIWAY` accepts only explicitly disclosed finite-game approximations and never falls back to Claude; strict `GTO` rejects them. | Validate the complete live workflow against a real solver and the owned-simulator corpus. |
| Capture milestone 0 | A persistent MSS source, local-CV frame signatures, change/peak/stable keyframes, bounded ACK-aware storage, and an immutable gap ledger are implemented. The observer-only sampler makes no model call. | Production calibration and long-running overflow/failure measurements remain. |
| Capture milestone 1 bridge | A provider-neutral sequential worker consumes atomic keyframe/gap batches, requires explicit observation/transient/gap decoder outcomes, feeds the transactional coordinator, and ACKs only after commit or fail-closed invalidation. Decoder timeouts poison the worker instead of accumulating calls. The application now owns its consumer-before-producer lifecycle. | Prove shutdown, crash recovery, and sustained operation on the production table layout. |
| Capture milestones 2–3 | The default-off Gemini decoder makes one bounded request per stable frame, normalizes capture identity/time, recognizes only strict new-hand anchors, and publishes an atomic accepted frame/snapshot/history bundle. Any capture/decode/queue gap or ambiguity revokes v3 readiness. | Prove gap-free preflop-to-river reconstruction on owned-simulator fixtures, including genuine 3–6-way hands. Visually unproved opponent checks remain intentional gaps. |
| Actual solver | A synthetic adapter can exercise the boundary. | The actual Monker request mapper, lifecycle controller, result decoder, policy/EV extractor, and convergence-proof mapper do not exist yet. |

## Monker integration unknowns

Monker is a plausible local multiway candidate, but the public workflow is
GUI-oriented and does not establish a supported server API. Before treating it
as the backend, obtain written vendor or license confirmation for:

1. Headless, CLI, file-based, or otherwise supported automation, including
   remote-server/RDP activation rights and allowed machine count.
2. Stable machine-readable import/export formats for full trees and individual
   decision nodes, not only displayed ranges.
3. Complete action policy, action amounts, EVs, iteration count, stopping
   metric, and convergence target. The adapter must not fabricate fields that
   Monker does not expose.
4. Supported player counts and complete semantics for stacks, blinds, antes,
   rake, all-ins, side pots, uncalled bets, off-tree sizes, folded-card
   bunching, and card abstraction.
5. Whether one action-conditioned state can be retained or losslessly restored
   from preflop through river, including checkpoint and cache behavior.
6. CPU/thread/NUMA scaling, peak memory, disk footprint, determinism, and
   version-to-version file compatibility for the proposed tree.
7. The precise multiplayer solution concept and meaning of the reported
   convergence value.

If only GUI automation is available, treat it as a fragile offline experiment,
not a production API. If convergence or complete policy data cannot be
exported, the correct v3 result is unsupported/failed—not a synthetic solved
proof.

## Expected workflow

1. The Mac samples calibrated table regions and retains transition keyframes.
2. The experimental decoder converts stable keyframes to ordered canonical
   public actions and board deals. The worker ACKs frames only
   after transactional semantic consumption or explicit fail-closed
   invalidation; any gap invalidates the hand. It never invents an invisible
   check or ambiguous hand boundary.
3. `PublicHandEventRecorder` freezes a gap-free `PublicHandHistory`.
4. At Hero's turn, the current screen state is cross-checked against replay and
   a `MultiwayDecisionState` is fingerprinted.
5. The v3 client sends the request over authenticated HTTPS while pinning the
   expected capability manifest.
6. The server replays the transcript, applies per-state capability gates, and
   invokes the external adapter.
7. **Remaining adapter/decoder:** the adapter restores or constructs the exact
   declared solver node, queries Monker, and emits the complete structured
   policy and proof.
8. Server and client verify request identity, decision fingerprint, backend and
   manifest identity, convergence, frequency conservation, legal actions, and
   all-in/minimum-raise targets.
9. The Mac renders the validated policy, then performs the existing final
   freshness recapture before showing advice.

For live latency, prefer a one-shot protocol shim connected to a persistent
local solver daemon. Starting and rebuilding a large Monker tree for every Hero
decision is unlikely to meet the deadline.

## Experimental target environment (still requires a real solver adapter)

Mac client:

```dotenv
STRATEGY_BACKEND=GTO_MULTIWAY
GTO_LIVE_ENABLED=1
GTO_OWNED_SIMULATOR_ACK=1
GTO_EXECUTION_MODE=remote

GTO_REMOTE_ENABLED=1
GTO_REMOTE_PROTOCOL=multiway-v3
GTO_REMOTE_ENDPOINT=https://solver.example.com/v1/evaluate
GTO_REMOTE_AUTH_TOKEN=replace-with-a-private-32-byte-or-longer-token
GTO_REMOTE_CAPABILITIES_PATH=/absolute/path/to/audited-capabilities.json
GTO_REMOTE_TIMEOUT_SECONDS=300
GTO_REMOTE_ALLOW_INSECURE_HTTP=0

# Default-off semantic capture; not production-ready until the capture
# acceptance corpus passes. This starts capture without LIVE_CAPTURE_ENABLED.
PUBLIC_HISTORY_DECODER_ENABLED=1
PUBLIC_HISTORY_DECODE_TIMEOUT_SECONDS=8
LIVE_CAPTURE_FPS=8
LIVE_CAPTURE_MAX_PENDING=128
LIVE_CAPTURE_HEARTBEAT_SECONDS=5
```

Solve server:

```dotenv
GTO_LIVE_ENABLED=1
GTO_OWNED_SIMULATOR_ACK=1
GTO_REMOTE_AUTH_TOKEN=the-same-private-token-as-the-Mac
GTO_SERVER_BACKEND=external

GTO_EXTERNAL_COMMAND_JSON=["/opt/ranges-adapter/bin/ranges-adapter","evaluate"]
GTO_EXTERNAL_CAPABILITIES_PATH=/opt/ranges-adapter/capabilities.json
GTO_EXTERNAL_WORKDIR=/opt/ranges-adapter
GTO_EXTERNAL_ENV_JSON={"LICENSE_FILE":"/opt/monker/license","DAEMON_SOCKET":"/run/ranges-adapter.sock"}
GTO_EXTERNAL_TIMEOUT_SECONDS=300
GTO_EXTERNAL_MAX_REQUEST_BYTES=4194304
GTO_EXTERNAL_MAX_RESPONSE_BYTES=1048576
GTO_EXTERNAL_ALLOW_BEST_EFFORT_PROCESS_CLEANUP=0
```

`GTO_REMOTE_CAPABILITIES_PATH` and
`GTO_EXTERNAL_CAPABILITIES_PATH` may be different filesystem paths, but they
must contain the same audited declaration and produce the same manifest
fingerprint. Do not pass the remote bearer token or general application
environment to the adapter through `GTO_EXTERNAL_ENV_JSON`.

The HTTP parser recognizes schema v2/v3, but dispatch is backend-dependent:
the bundled native router advertises only v2, while the external adapter
advertises v2 and v3. `GTO_REMOTE_PROTOCOL` is the Mac-side selection. A
manifest may declare `SOLVED_TREE`, `MULTIWAY_TREE`, six players, stateful
range conditioning, bunching, card-exact private cards, an action model, and a
convergence metric only after each statement is verified. A fixed or dynamic
discrete action tree is always an explicit finite-game approximation;
`CONTINUOUS_NO_LIMIT` is required before the current validator permits an exact
action-space claim. Hardware size is not evidence for any capability.

## Hardware pilot and scaling

Start with one rented **128 GiB** host after the licensing/automation questions
are answered. Prefer 16–32 strong CPU cores, fast local NVMe, and an OS supported
by the licensed build. Keep the API single-flight and reserve memory for the
OS, server, adapter, solver runtime, tree files, and cache. For an external
solver, use its own memory setting plus a service/cgroup ceiling;
`GTO_ENGINE_MAX_MEMORY_GIB` controls the bundled native engine, not Monker.

Run the exact proposed tree and record:

- native estimated memory versus observed peak and steady RSS;
- build/load/checkpoint time, iterations per hour, and time to the declared
  target;
- cold and warm decision latency at every street;
- CPU utilization and scaling by thread count;
- disk/cache size and restart recovery;
- repeated-request memory growth and failure behavior;
- policy and proof reproducibility for a fixed corpus.

Choose the next tier from measurements, not vendor examples. A useful planning
rule is:

```text
required physical RAM = measured p99 peak solver RSS × 1.30
                      + at least 16 GiB service/OS headroom
```

- Stay at 128 GiB when the representative corpus fits without swap/OOM and
  meets the latency/convergence gate.
- Move to 256 GiB when the measured requirement exceeds the safe 128 GiB
  envelope or the accepted tree cannot be built there.
- Move to 512 GiB only when the same measurement exceeds the safe 256 GiB
  envelope, or a separately approved offline batch design requires it. The
  current live server runs one solve at a time, so unused concurrency is not a
  reason by itself.

More RAM only permits a larger retained tree. It does not create an automation
API, make the experimental capture decoder production-ready, provide truthful
bunching or a convergence certificate, or accelerate single-thread CPU
convergence by itself.

For a concrete order-of-magnitude budget, Hetzner's published 15 June 2026
dedicated-server price table lists, excluding IPv4 and before VAT/Windows
licensing:

| Host | RAM | Monthly | Setup |
| --- | ---: | ---: | ---: |
| AX102-1 | 128 GiB | EUR 257.30 | EUR 129 |
| AX162-1 | 128 GiB ECC | EUR 612.30 | EUR 304 |
| AX162-2 | 256 GiB ECC | EUR 842.30 | EUR 419 |
| AX162-3 | 512 GiB ECC | EUR 1,597.30 | EUR 799 |

Source: [Hetzner price adjustment](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)
and [AX hardware configurations](https://docs.hetzner.com/robot/dedicated-server/server-lines/ax-server/).
Recheck availability, tax, operating-system licensing, IPv4, and current price
at order time. The economical first experiment is therefore one month on the
AX102-1 class, not an immediate 256/512-GiB commitment.

## Acceptance gates

Promotion to live `GTO_MULTIWAY`, under the explicit
`APPROXIMATE_SOLVER`/finite-abstraction label, requires every gate:

1. **Legal/vendor:** automation, server activation, and export rights are
   documented for the deployed version.
2. **Capture:** known owned-simulator hands reconstruct every action from the
   initial preflop state through river; overflow, ambiguous transition, or
   decoder failure prevents solving. Include genuine 3–6-way hands, folds to
   heads-up, all-ins, short raises, and side pots.
3. **Contracts:** protocol, outcome, client, external-backend, server, and
   capture suites pass, including malformed/oversized inputs, identity
   mismatches, illegal policies, non-convergence, and approximation disclosure.
4. **Adapter fidelity:** every request field maps to an audited Monker setting;
   every response field comes from a documented artifact. Golden fixtures
   match manual Monker inspection.
5. **Game coverage:** preflop through river, rake/ante profiles, stack depths,
   all-in/side-pot handling, off-tree policy, bunching, card model, and action
   abstraction are versioned and covered by a representative corpus.
6. **Proof truthfulness:** the multiplayer metric and solution concept are
   defined; `metric_value <= target_value`; backend/version/manifest/profile/
   abstraction identities are pinned. Any abstraction gap is disclosed with
   `approximate=true` and is rejected by strict `GTO`.
7. **Independent check:** sampled policies, EVs, and convergence values are
   compared with an independent export, manual solver view, or second licensed
   implementation—not merely the chosen dominant action.
8. **128 GiB pilot:** peak memory, cold/warm latency, convergence, restart,
   timeout, crash, and leak tests pass. Scaling to 256/512 follows the measured
   rule above.
9. **Operations:** TLS, private token handling, loopback solver service,
   bounded logs, daemon health, idempotency, cache invalidation by backend/
   profile/abstraction version, and final screen freshness all fail closed.

Promotion beyond that to strict `GTO` additionally requires zero exactness
gaps, a `CONTINUOUS_NO_LIMIT` action model, and a proof with zero metric value.
A useful converged finite tree normally does not meet those stronger
conditions and must remain in `GTO_MULTIWAY`.

Until these gates pass, the correct operational status is “integration
boundary ready; multiway solver unavailable.” The narrower bundled heads-up
backend remains truthful, and a rented larger machine must not change that
label.
