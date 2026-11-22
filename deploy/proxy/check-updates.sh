#!/usr/bin/env bash
# Poll ghcr for new space images; run redeploy.sh if any digest changed.
# Exit codes: 0 no change, 10 redeployed, 20 error.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${SPACE_TAG:-latest}"
IMAGES=(
  "ghcr.io/eliasstepanik/space-shard:${TAG}"
  "ghcr.io/eliasstepanik/space-orchestrator:${TAG}"
  "ghcr.io/eliasstepanik/space-client:${TAG}"
)

log() { echo "[$(date -u +%FT%TZ)] check-updates: $*"; }

changed=0
for img in "${IMAGES[@]}"; do
  if ! out="$(docker buildx imagetools inspect "$img" 2>&1)"; then
    if grep -qiE 'not found|denied|unauthorized|401|404|manifest unknown|name unknown' <<<"$out"; then
      log "$img: not available yet (not published or not public); skipping"
      continue
    fi
    log "ERROR inspecting $img: $out"
    exit 20
  fi
  remote="$(awk '/^Digest:/ {print $2; exit}' <<<"$out")"
  if [ -z "$remote" ]; then
    log "ERROR: no digest in imagetools output for $img"
    exit 20
  fi
  # RepoDigests entries look like ghcr.io/owner/name@sha256:...
  local_digests="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$img" 2>/dev/null || true)"
  if grep -qF "@${remote}" <<<"$local_digests"; then
    log "$img: up to date ($remote)"
  else
    log "$img: remote $remote differs from local"
    changed=1
  fi
done

if [ "$changed" -eq 0 ]; then
  exit 0
fi
log "change detected; running redeploy.sh"
"$DIR/redeploy.sh" || { log "ERROR: redeploy.sh failed"; exit 20; }
exit 10
