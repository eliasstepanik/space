# Space CI build image: runs the full project CI (`cargo xtask ci`: fmt,
# clippy -D warnings, tests) in a container, plus the wasm viewer build.
#
#   docker build -t space/ci .          # runs CI as a build step; fails if CI fails
#
# NOT EXECUTED: the authoring sandbox has no Docker daemon. Correct by
# inspection only (deploy/README.md). The authoritative CI is
# .github/workflows/ci.yml and `cargo xtask ci`.
FROM rust:1-slim-bookworm
WORKDIR /src
# rust-toolchain.toml pins the toolchain (1.100.0 + rustfmt, clippy, wasm32 target);
# `rustup toolchain install` with no argument installs exactly that. build-essential
# (a C compiler/linker) is for `ring`, the one sanctioned C/asm dependency.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY rust-toolchain.toml .
RUN rustup toolchain install
COPY . .
# Full CI (GPU tests skip themselves when no Vulkan device is present).
RUN cargo run -p xtask -- ci
# The wasm viewer builds for the pinned wasm32 target.
RUN cargo build --release --locked --target wasm32-unknown-unknown -p space-viewer
