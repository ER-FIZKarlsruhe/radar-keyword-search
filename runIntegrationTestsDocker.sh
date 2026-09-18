#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="radar-keyword-search-integration-tests"

# Some agents set HTTP_PROXY/HTTPS_PROXY with a stray trailing comma
# (e.g. "http://proxy:8888,"), which pip's URL parser rejects. Strip it.
# NO_PROXY is left untouched since commas there are a valid domain separator.
HTTP_PROXY_CLEAN="${HTTP_PROXY:-}"
HTTP_PROXY_CLEAN="${HTTP_PROXY_CLEAN%,}"
HTTPS_PROXY_CLEAN="${HTTPS_PROXY:-}"
HTTPS_PROXY_CLEAN="${HTTPS_PROXY_CLEAN%,}"

# The Ollama server is started by testcontainers *from inside* this
# container, using the host's Docker daemon (Docker-outside-of-Docker) via
# the socket mounted below - the Ollama container ends up as a sibling of
# this one, not nested inside it. Its model cache is bind-mounted by
# absolute path, and Docker always resolves bind-mount sources against the
# real host, not the container that requested them. So this wrapper
# container must mount the cache dir at that exact same absolute path,
# otherwise Docker would just create an empty directory tree on the host at
# whatever path our process happens to compute, and the cache would never
# be reused. OLLAMA_TEST_CACHE_DIR is passed through to Python explicitly so
# both sides agree on the very same path regardless of what $HOME resolves
# to inside vs. outside the image.
OLLAMA_CACHE_DIR="${OLLAMA_TEST_CACHE_DIR:-$HOME/.cache/radar-keyword-search-ollama-test}"
mkdir -p "$OLLAMA_CACHE_DIR"

docker build \
  --build-arg HTTP_PROXY="${HTTP_PROXY_CLEAN:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY_CLEAN:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  -f Dockerfile.integration \
  -t "$IMAGE_TAG" \
  .

docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$OLLAMA_CACHE_DIR:$OLLAMA_CACHE_DIR" \
  -e OLLAMA_TEST_CACHE_DIR="$OLLAMA_CACHE_DIR" \
  "$IMAGE_TAG"
