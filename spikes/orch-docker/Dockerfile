# Phase-0 spike 0.6: orchestrator image.
#
# Multi-stage: build the standalone `orch-docker` binary, then ship it on a
# minimal runtime that can talk to the mounted Docker socket.
#
# NOTE: this is a throwaway spike image. Phase 8 will refine the real
# `space-orchestrator` image (distroless / hardened, no docker group).

# ---- build stage ----------------------------------------------------------
FROM rust:1.100-bookworm AS build
WORKDIR /src
# Copy only the standalone spike package (it has its own empty [workspace]).
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Release build; the spike is standalone so we build in its own dir.
RUN cargo build --release

# ---- runtime stage --------------------------------------------------------
FROM debian:bookworm-slim AS runtime
# `curl` is only for a possible healthcheck; `ca-certificates` for any TLS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user. The compose file adds this user to the `docker` group so it
# can access the mounted /var/run/docker.sock (group id is matched at runtime).
RUN useradd --create-home --uid 10001 orchestrator
COPY --from=build /src/target/release/orch-docker /usr/local/bin/orch-docker
USER orchestrator

ENV HOST_PORT_RANGE=46001-46064
ENTRYPOINT ["/usr/local/bin/orch-docker"]
