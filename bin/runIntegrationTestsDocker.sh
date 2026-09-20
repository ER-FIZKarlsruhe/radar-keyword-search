#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from,
# so the Docker build context and test-results output always land right.
cd "$(dirname "$0")/.."

# On Windows, Git Bash auto-converts anything that looks like a Unix path
# into a Windows path before handing it to docker.exe - including the
# *container-side* half of a `-v host:container` flag below, which must
# stay a literal Linux path. Left unset, that mangles the bind mounts so
# nothing actually lands in ./test-results on the host. No-op on Linux
# (e.g. the real Bamboo agent), so always safe to set.
export MSYS_NO_PATHCONV=1

IMAGE_TAG="radar-keyword-search-integration-tests"
OLLAMA_IMAGE="ollama/ollama:0.1.44"
MODEL_NAME="qwen2.5:0.5b"

# Unique per invocation so concurrent builds on the same Bamboo agent don't
# collide, and so a stale name from a previous crashed run never conflicts.
RUN_ID="$$"
NETWORK_NAME="radar-keyword-search-ollama-test-net-${RUN_ID}"
OLLAMA_CONTAINER_NAME="radar-keyword-search-ollama-test-server-${RUN_ID}"

# Some agents set HTTP_PROXY/HTTPS_PROXY with a stray trailing comma
# (e.g. "http://proxy:8888,"), which pip's URL parser rejects. Strip it.
# NO_PROXY is left untouched since commas there are a valid domain separator.
HTTP_PROXY_CLEAN="${HTTP_PROXY:-}"
HTTP_PROXY_CLEAN="${HTTP_PROXY_CLEAN%,}"
HTTPS_PROXY_CLEAN="${HTTPS_PROXY:-}"
HTTPS_PROXY_CLEAN="${HTTPS_PROXY_CLEAN%,}"

OLLAMA_CACHE_DIR="${OLLAMA_TEST_CACHE_DIR:-$HOME/.cache/radar-keyword-search-ollama-test}"
mkdir -p "$OLLAMA_CACHE_DIR"

# The official ollama/ollama image sets OLLAMA_HOST=0.0.0.0 so the *server*
# binds all interfaces - but `ollama pull`, run as a separate client process
# via `docker exec`, also reads that same OLLAMA_HOST to know where to
# connect *to*. With HTTP_PROXY/HTTPS_PROXY set, that local client->server
# call gets routed through the proxy too unless NO_PROXY exempts it - and
# it's "0.0.0.0" specifically that needs the exemption, not "localhost" or
# "127.0.0.1" (confirmed by reproducing this locally: those two alone still
# failed with "could not connect to ollama app, is it running?"; adding
# 0.0.0.0 fixed it). Keep all three so this doesn't quietly break again if
# the image's OLLAMA_HOST default ever changes.
OLLAMA_NO_PROXY="0.0.0.0,localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"

cleanup() {
  docker rm -f "$OLLAMA_CONTAINER_NAME" >/dev/null 2>&1 || true
  docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build \
  --build-arg HTTP_PROXY="${HTTP_PROXY_CLEAN:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY_CLEAN:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  -f docker/Dockerfile.integration \
  -t "$IMAGE_TAG" \
  .

# Ollama is started directly here (as a plain sibling container on the
# host), not from inside the test container via testcontainers. This avoids
# Docker-outside-of-Docker entirely for the test container - no docker.sock
# mount, and no separate proxy propagation into a container started through
# the raw Docker Engine API (which never picks up the corporate proxy the
# way `docker run`/`docker build` do here).
docker network create "$NETWORK_NAME" >/dev/null

docker run -d \
  --name "$OLLAMA_CONTAINER_NAME" \
  --network "$NETWORK_NAME" \
  -e HTTP_PROXY="${HTTP_PROXY_CLEAN:-}" \
  -e HTTPS_PROXY="${HTTPS_PROXY_CLEAN:-}" \
  -e NO_PROXY="$OLLAMA_NO_PROXY" \
  -e OLLAMA_DEBUG=1 \
  -v "$OLLAMA_CACHE_DIR:/root/.ollama" \
  "$OLLAMA_IMAGE" >/dev/null

echo "Waiting for the Ollama server to start..."
started=0
for _ in $(seq 1 30); do
  if docker logs "$OLLAMA_CONTAINER_NAME" 2>&1 | grep -q "Listening on "; then
    started=1
    break
  fi
  sleep 1
done
if [ "$started" -ne 1 ]; then
  echo "Ollama server did not start in time - server logs:" >&2
  docker logs "$OLLAMA_CONTAINER_NAME" >&2
  exit 1
fi

echo "Pulling model ${MODEL_NAME}..."
if ! docker exec "$OLLAMA_CONTAINER_NAME" ollama pull "$MODEL_NAME"; then
  echo "'ollama pull ${MODEL_NAME}' failed - ollama server logs:" >&2
  docker logs "$OLLAMA_CONTAINER_NAME" >&2
  exit 1
fi

mkdir -p test-results

docker run --rm \
  --network "$NETWORK_NAME" \
  -e EXTRACTION_BACKEND=keyllm \
  -e "OLLAMA_BASE_URL=http://${OLLAMA_CONTAINER_NAME}:11434/v1" \
  -e OLLAMA_MODEL="$MODEL_NAME" \
  -v "$(pwd)/test-results:/app/test-results" \
  "$IMAGE_TAG"
