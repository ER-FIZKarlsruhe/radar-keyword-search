#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="radar-keyword-search-tests"

docker build \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  -f Dockerfile.test \
  -t "$IMAGE_TAG" \
  .

docker run --rm "$IMAGE_TAG"
