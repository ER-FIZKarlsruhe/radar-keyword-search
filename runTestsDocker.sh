#!/usr/bin/env bash
set -euo pipefail

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
  -f Dockerfile.test \
  -t "$IMAGE_TAG" \
  .

docker run --rm "$IMAGE_TAG"
