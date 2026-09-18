#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from,
# so the Docker build context and test-results output always land right.
cd "$(dirname "$0")/.."

# On Windows, Git Bash auto-converts anything that looks like a Unix path
# into a Windows path before handing it to docker.exe - including the
# *container-side* half of a `-v host:container` flag below, which must
# stay a literal Linux path. Left unset, that mangles the bind mount so
# nothing actually lands in ./test-results on the host. No-op on Linux
# (e.g. the real Bamboo agent), so always safe to set.
export MSYS_NO_PATHCONV=1

IMAGE_TAG="radar-keyword-search-tests"

# Some agents set HTTP_PROXY/HTTPS_PROXY with a stray trailing comma
# (e.g. "http://proxy:8888,"), which pip's URL parser rejects. Strip it.
# NO_PROXY is left untouched since commas there are a valid domain separator.
HTTP_PROXY_CLEAN="${HTTP_PROXY:-}"
HTTP_PROXY_CLEAN="${HTTP_PROXY_CLEAN%,}"
HTTPS_PROXY_CLEAN="${HTTPS_PROXY:-}"
HTTPS_PROXY_CLEAN="${HTTPS_PROXY_CLEAN%,}"

docker build \
  --build-arg HTTP_PROXY="${HTTP_PROXY_CLEAN:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY_CLEAN:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  -f docker/Dockerfile.test \
  -t "$IMAGE_TAG" \
  .

mkdir -p test-results

docker run --rm -v "$(pwd)/test-results:/app/test-results" "$IMAGE_TAG"
