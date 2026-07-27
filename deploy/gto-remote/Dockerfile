# syntax=docker/dockerfile:1

FROM rust:1.97.0-bookworm AS engine-builder
WORKDIR /build/gto_oracle_engine
COPY gto_oracle_engine/Cargo.toml gto_oracle_engine/Cargo.lock ./
COPY gto_oracle_engine/.cargo ./.cargo
COPY gto_oracle_engine/src ./src
RUN cargo build --release --locked

FROM python:3.12-slim-bookworm
RUN groupadd --gid 10001 gto-oracle \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin gto-oracle \
    && mkdir -p /app /data/cache /data/preflop_blueprint_cache \
    && chown -R gto-oracle:gto-oracle /data

WORKDIR /app
COPY --from=engine-builder \
    /build/gto_oracle_engine/target/release/gto-oracle-engine \
    /usr/local/bin/gto-oracle-engine
COPY gto_oracle ./gto_oracle
COPY gto_remote ./gto_remote
COPY gto_hand_history.py live_gto.py preflop_blueprint.py \
    preflop_history.py preflop_observation.py ./
COPY poker_data.json ./
COPY gto_oracle_engine/LICENSE gto_oracle_engine/NOTICE.md \
    /usr/share/doc/gto-oracle-engine/

ENV PYTHONUNBUFFERED=1 \
    GTO_SERVER_BASE_DIR=/app \
    GTO_API_BIND=0.0.0.0 \
    GTO_API_ALLOW_NONLOOPBACK=1 \
    GTO_API_PORT=8787 \
    GTO_LIVE_ENABLED=1 \
    GTO_OWNED_SIMULATOR_ACK=1 \
    GTO_ENGINE_PATH=/usr/local/bin/gto-oracle-engine \
    GTO_CACHE_PATH=/data/cache/gto_live_cache.sqlite3 \
    PREFLOP_BLUEPRINT_CACHE_PATH=/data/preflop_blueprint_cache \
    PREFLOP_BLUEPRINT_ALLOW_NETWORK=0

USER 10001:10001
EXPOSE 8787
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python3", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8787/health/ready', timeout=3).read()"]
CMD ["python3", "-m", "gto_remote.server"]
