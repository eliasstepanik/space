#!/usr/bin/env bash
# Pull the newest space images and (re)start the stack. Idempotent: `up -d` only
# recreates containers whose image changed. Logs to stdout (journald when run
# from the systemd timer; the SSH output in the manual GitHub Actions path).
#
# Serialized with flock so the poll timer and a workflow_dispatch run never
# overlap. Runnable from any cwd: everything uses absolute paths.
set -euo pipefail

COMPOSE_FILE="${SPACE_COMPOSE_FILE:-/opt/space/docker-compose.yaml}"
LOCKFILE="${SPACE_LOCKFILE:-/var/lock/space-redeploy.lock}"

log() { echo "[$(date -u +%FT%TZ)] redeploy: $*"; }

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  log "another redeploy is running; waiting for the lock"
  flock 9
fi

log "pulling images"
# --profile '*' also pulls the stream client image when its profile is off, so
# the digest checks in check-updates.sh do not flap.
docker compose -f "$COMPOSE_FILE" --profile '*' pull --quiet
log "up -d"
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans
log "pruning dangling images"
docker image prune -f
log "done"
