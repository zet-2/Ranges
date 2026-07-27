# Remote live-GTO evaluator

This package keeps OCR and table capture on the Mac while moving range
reconstruction, cache access, and the CPU/RAM-heavy solve to a Linux machine.
The Mac sends a strict `LiveDecisionState`; it never attempts to construct a
remote `SolveSpec`.

The live path is:

```text
Mac capture/OCR
  -> local event recorder, legality, and freshness checks
  -> authenticated LiveDecisionState + gap-free public-hand JSON
  -> remote native router or owned/licensed external solver adapter
  -> minimal LiveGTOOutcome JSON
  -> local legality check and final screen recapture
```

Screenshots, Gemini/Anthropic credentials, the Hero username, and the Mac's
general `.env` are not part of the wire schema.

The `public_hand` field is present only after the Mac has observed the untouched
preflop forced-bet frame and every subsequent public transition. It records
2–6 occupied seats, starting stacks, blinds, ante, rake profile, folds, checks,
calls, bet-to/raise-to/all-in targets, and board deals. The server independently
replays it with action-order, minimum-raise, stack, pot, board, and chip-
conservation checks. A declared full six-max backend refuses a request without
this transcript; sparse screenshots are never converted into invented actions.
The native HU backend also consumes it whenever the hand was heads-up at the
flop: it reconstructs the true flop tree, conditions both ranges on every
recorded action/card through river, and caches the resulting exact path.

## HTTP contract

`POST /v1/evaluate` is synchronous and accepts:

```json
{
  "schema_version": 2,
  "request_id": "13de5c6d-919c-4be5-8ef2-e6cf08a3404f",
  "state": {
    "...": "serialized by gto_remote.protocol"
  }
}
```

Required headers:

```text
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: <same value as request_id>
X-Request-ID: <same value as request_id>
```

The response contains only the decision fingerprint and the small
`LiveGTOOutcome` surface: status, reason, latency, analysis, source, model,
cache hit, approximation flag, and solver spec key. Full ranges and result
matrices remain on the server.

Only one evaluation can use the router at a time. A concurrent request gets
HTTP `429` with `Retry-After: 1`; the client should not start an unbounded retry
loop. Recent completed responses are replayed in process memory when the same
request ID and decision fingerprint are submitted again. Reusing an ID for a
different state returns HTTP `409`.

The no-network end-to-end harness exercises the same protocol and service
boundary:

```sh
.venv/bin/python gto_full_process_simulation.py --mode dry-run --street RIVER
.venv/bin/python gto_full_process_simulation.py \
  --mode real --street RIVER --target 0.1 --max-iterations 100000
```

It verifies that the range provider receives only the transcript through
`DEAL_FLOP`; turn/river cards enter only through solver chance nodes.

Health endpoints:

- `GET /health/live`: the HTTP process is alive.
- `GET /health/ready`: GTO is enabled, the owned-simulator acknowledgement is
  set, and the configured backend executable is available. It also reports
  `backend_id` and the derived `full_six_max_ready` value.
- `GET /v1/about`: authenticated protocol/license/source information plus the
  complete solver capability manifest and every remaining six-max gap.

## Mac client configuration

After the solve server has a TLS URL, configure the Mac's private `.env`:

```dotenv
STRATEGY_BACKEND=GTO
GTO_LIVE_ENABLED=1
GTO_OWNED_SIMULATOR_ACK=1
GTO_EXECUTION_MODE=remote

GTO_REMOTE_ENABLED=1
GTO_REMOTE_ENDPOINT=https://solver.example.com/v1/evaluate
GTO_REMOTE_AUTH_TOKEN=the-same-private-token-as-the-server
GTO_REMOTE_TIMEOUT_SECONDS=300
GTO_REMOTE_ALLOW_INSECURE_HTTP=0
```

`GTO_ENGINE_PATH`, the solver cache, and the blueprint cache are server-owned
in remote mode. A bad endpoint, timeout, authentication failure, oversized
body, mismatched request ID, mismatched state fingerprint, or malformed
response becomes a failed GTO outcome; strict `GTO` does not fall back to a
language model.

For initial testing without a domain, keep the server on loopback and open an
SSH tunnel:

```sh
ssh -N -L 8787:127.0.0.1:8787 solver@SERVER_IP
```

Then use:

```dotenv
GTO_REMOTE_ENDPOINT=http://127.0.0.1:8787/v1/evaluate
GTO_REMOTE_ALLOW_INSECURE_HTTP=1
```

Plain HTTP is accepted by the client only for a loopback URL and only after
that explicit switch. The SSH connection provides encryption in this layout.

RunPod's
[managed HTTP proxy](https://docs.runpod.io/pods/configuration/expose-ports)
currently limits a request to 100 seconds. This API is synchronous and the
configured unseen-flop budget is 180 seconds, so do not route long cold solves
through that proxy. Use full SSH port forwarding, a direct TLS endpoint, or
pre-warm the cache and keep the solve deadline below the proxy limit. The
dedicated-server Caddy layout has no such provider proxy deadline.

## Server environment

Generate a private token with at least 32 bytes, for example:

```sh
openssl rand -hex 32
```

Minimum server configuration:

```sh
export GTO_REMOTE_AUTH_TOKEN='<generated token>'
export GTO_SERVER_BASE_DIR=/opt/gto-oracle
export GTO_LIVE_ENABLED=1
export GTO_OWNED_SIMULATOR_ACK=1
export GTO_SERVER_BACKEND=native
export GTO_ENGINE_PATH=/opt/gto-oracle/bin/gto-oracle-engine
export GTO_CACHE_PATH=/var/lib/gto-oracle/gto_live_cache.sqlite3
export PREFLOP_BLUEPRINT_CACHE_PATH=/opt/gto-oracle/preflop_blueprint_cache
export GTO_ENGINE_MAX_MEMORY_GIB=96
export GTO_API_SOURCE_URL='https://your-source-host/repository-at-deployed-revision'
python3 -m gto_remote.server
```

`native` is the bundled backend: fixed six-max preflop blueprint plus a
heads-up postflop subgame solver. Its manifest deliberately reports that it is
not full six-max. Hardware cannot change those declared game-model limits.

### External full-solver adapter

`GTO_SERVER_BACKEND=external` replaces the native router without changing the
Mac protocol. It is a strict integration boundary for a solver executable that
you own or are licensed to automate:

```dotenv
GTO_SERVER_BACKEND=external
GTO_EXTERNAL_COMMAND_JSON=["/opt/full-gto/bin/ranges-adapter","evaluate"]
GTO_EXTERNAL_CAPABILITIES_PATH=/opt/full-gto/capabilities.json
GTO_EXTERNAL_WORKDIR=/opt/full-gto
GTO_EXTERNAL_ENV_JSON={"LICENSE_FILE":"/opt/full-gto/license.key"}
GTO_EXTERNAL_TIMEOUT_SECONDS=300
GTO_EXTERNAL_MAX_REQUEST_BYTES=4194304
GTO_EXTERNAL_MAX_RESPONSE_BYTES=1048576
```

The command is executed directly, never through a shell. It reads exactly one
protocol-v2 evaluate request from stdin and writes exactly one protocol-v2
response to stdout. Request ID and state fingerprint must match. Timeout,
nonzero exit, oversized output, malformed JSON, or an identity mismatch fails
closed. The server bearer token and API-provider keys are not inherited by the
adapter; only `PATH`, locale/timezone values, and the explicit
`GTO_EXTERNAL_ENV_JSON` mapping are supplied.

The manifest schema is demonstrated by
[`external-capabilities.not-ready.example.json`](../deploy/gto-remote/external-capabilities.not-ready.example.json).
The example is intentionally not ready. Replace every declaration with audited
facts from the integrated backend. A full-ready declaration requires a
compatible solved preflop tree, multiway postflop support through six players,
action-conditioned continuation through river, folded-card bunching, a
card-exact private-card model, and a real convergence metric. The server derives
the readiness flag and gaps again and rejects contradictory manifests.

The one-shot adapter may be a thin client to a persistent local solver daemon.
It must not claim `stateful_through_river=true` unless it actually retains or
losslessly reconstructs the same action-conditioned game state across streets.
No proprietary solver is bundled, and this interface is not evidence that a
commercial GUI exposes an automatable API.

The default listener is `127.0.0.1:8787`. Keep it on loopback behind a TLS
reverse proxy. A non-loopback bind is rejected unless
`GTO_API_ALLOW_NONLOOPBACK=1` is explicitly set; that switch does not provide
TLS by itself.

Once started, check it before enabling the Mac:

```sh
curl --fail https://solver.example.com/health/live
curl --fail https://solver.example.com/health/ready
curl --fail \
  -H 'Authorization: Bearer YOUR_PRIVATE_TOKEN' \
  https://solver.example.com/v1/about
```

The Rust engine enforces `GTO_ENGINE_MAX_MEMORY_GIB` per solve. The API fixes
concurrency at one, so the host still needs that amount plus operating-system,
Python, SQLite, and tree-building headroom. Set a cgroup/systemd `MemoryMax`
above the engine guard, not equal to it.

Suggested starting guards:

| Physical RAM | `GTO_ENGINE_MAX_MEMORY_GIB` | systemd `MemoryMax` |
| ---: | ---: | ---: |
| 128 GiB | 96 | `112G` |
| 256 GiB | 192 | `224G` |
| 384 GiB | 288 | `336G` |

These are safety ceilings, not a promise that every tree below the ceiling
will converge within the live deadline.

## Deployment

Templates are in `deploy/gto-remote`:

- `gto-remote.service` runs the server as an unprivileged systemd service.
- `Caddyfile` terminates public HTTPS and proxies to loopback.
- `Dockerfile` builds the pinned Rust engine and a non-root Python runtime.
- `server.env.example` documents the native systemd environment.
- `container.env.example` contains paths and limits for the container image.
- `external-capabilities.not-ready.example.json` is a fail-closed capability
  manifest template for a future licensed adapter.

For a rented bare-metal solve node, systemd plus Caddy is the simplest layout.
Allow inbound ports 80/443 only, and keep port 8787 firewalled. Do not copy the
Mac's general `.env`; provision only the remote token and GTO variables.
Keep `/var/lib/gto-oracle` (or Docker's `/data`) on persistent NVMe storage so
solver results survive restarts. The preflop blueprint cache must also be
populated on the server before network fetching is disabled.

An exact container rollout from the repository root is:

```sh
docker build \
  -f deploy/gto-remote/Dockerfile \
  -t ranges-gto:local .

sudo install -d -m 0700 /etc/gto-oracle
sudo install -m 0600 \
  deploy/gto-remote/container.env.example \
  /etc/gto-oracle/container.env
sudo install -d -o 10001 -g 10001 -m 0700 /var/lib/gto-oracle

docker run --rm \
  --user 10001:10001 \
  -v /var/lib/gto-oracle:/data \
  ranges-gto:local \
  python3 preflop_blueprint.py sync \
    --cache-dir /data/preflop_blueprint_cache \
    --stack 100 \
    --max-depth 4 \
    --workers 8

docker run --rm \
  --user 10001:10001 \
  -v /var/lib/gto-oracle:/data \
  ranges-gto:local \
  python3 preflop_blueprint.py validate \
    --cache-dir /data/preflop_blueprint_cache \
    --stack 100

docker run -d \
  --name ranges-gto \
  --restart unless-stopped \
  --env-file /etc/gto-oracle/container.env \
  --memory 224g \
  --pids-limit 512 \
  -p 127.0.0.1:8787:8787 \
  -v /var/lib/gto-oracle:/data \
  ranges-gto:local
```

Edit the two secrets, source URL, and memory limits before starting it. The
example uses the 256-GiB host profile. The depth-4 synchronization is the
complete published 100-BB tree and is a large one-time download. Replace
`solver.example.com` in the supplied Caddyfile, install it as
`/etc/caddy/Caddyfile`, and reload Caddy only after its configuration validates.

The engine links AGPL-covered software. When users interact with it over a
network, keep the exact corresponding source, dependency revision, build
instructions, license, and notice available. Set `GTO_API_SOURCE_URL` to that
deployed source revision and review `gto_oracle_engine/NOTICE.md`.

For extensive server-only validation, staged acceptance criteria, and cost
controls, see
[`docs/gto_server_validation.md`](../docs/gto_server_validation.md).
