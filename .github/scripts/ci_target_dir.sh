#!/usr/bin/env bash
# Give a job on the self-hosted runner a cargo target directory that survives
# the run, so the next run only rebuilds what changed.
#
#   bash .github/scripts/ci_target_dir.sh NAME
#
# actions/checkout cleans the workspace (`git clean -ffdx`) before every run,
# which deletes `target/` and makes every job a from-scratch build. This puts
# the target directory outside the workspace instead and exports it to the
# following steps as CARGO_TARGET_DIR (xtask and the dev scripts honour it).
#
# One directory per runner and NAME: a runner runs one job at a time, so no two
# builds ever share a directory, and it is safe to delete it here. Jobs that
# build the same thing pass the same NAME and share the directory (`wasm`: the
# CI wasm job and the release viewer job).
#
# The directory is deleted and started again when it has grown past the limit
# (SPACE_CI_CACHE_LIMIT_GB, default 25): stale artefacts of old dependency
# versions pile up, and the runner shares its disk with the production stack.
# SPACE_CI_CACHE moves the whole cache (default ~/.cache/space-ci).
#
# On a runner whose home directory does not survive the run this changes
# nothing: every run starts with an empty directory, as before.
set -euo pipefail

name="${1:?usage: ci_target_dir.sh NAME}"
case "$name" in
    *[!a-z0-9-]*) echo "ci_target_dir: NAME must be lowercase letters, digits and dashes" >&2; exit 2 ;;
esac
# The runner's name is free text; keep only what is safe in a directory name.
runner="$(printf '%s' "${RUNNER_NAME:-local}" | tr -c 'A-Za-z0-9._-' '_')"
base="${SPACE_CI_CACHE:-$HOME/.cache/space-ci}"
dir="$base/$runner/$name"
limit_gb="${SPACE_CI_CACHE_LIMIT_GB:-25}"

mkdir -p "$dir"
used_kb="$(du -sk "$dir" | cut -f1)"
if [ "$used_kb" -gt $((limit_gb * 1024 * 1024)) ]; then
    echo "ci_target_dir: $dir holds $((used_kb / 1024 / 1024)) GB (limit $limit_gb GB): starting again"
    rm -rf "$dir"
    mkdir -p "$dir"
    used_kb=0
fi
echo "ci_target_dir: CARGO_TARGET_DIR=$dir ($((used_kb / 1024)) MB from earlier runs)"

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "CARGO_TARGET_DIR=$dir"
        # Incremental compilation only helps when the same crate is rebuilt
        # after a small edit; here it would mostly add to the directory's size.
        echo "CARGO_INCREMENTAL=0"
    } >> "$GITHUB_ENV"
fi
