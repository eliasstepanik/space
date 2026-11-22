#!/bin/sh
# Pull the newest space images and (re)start the stack. Idempotent: `up -d` only
# recreates containers whose image changed. Logs to stdout (journald when run
# from the systemd timer; the SSH output in the manual GitHub Actions path).
set -eu

cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.yaml"

echo "[$(date -u +%FT%TZ)] redeploy: pulling images"
$COMPOSE pull --quiet
echo "[$(date -u +%FT%TZ)] redeploy: up -d"
$COMPOSE up -d --remove-orphans
echo "[$(date -u +%FT%TZ)] redeploy: pruning dangling images"
docker image prune -f
$COMPOSE ps
echo "[$(date -u +%FT%TZ)] redeploy: done"
