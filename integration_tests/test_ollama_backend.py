"""
End-to-end test of the Ollama backend against a REAL Ollama server.

Points iri_api at a real Ollama server and sends a real request through the
FastAPI app. This is slow (pulls a container image and a model on first run,
then runs real CPU inference) and requires Docker, so it's kept out of the
default `pytest` run - see requirements.txt in this directory for what's
needed to run it, and the "Ollama Integration Test" section in the README.

Run locally (starts and tears down its own Ollama container via testcontainers):
    pip install -r integration_tests/requirements.txt
    pytest integration_tests

Run via ../runIntegrationTestsDocker.sh (used by Bamboo): that script starts
the Ollama server itself and pulls the model into it, then sets
OLLAMA_BASE_URL before running pytest - see the ollama_backend fixture below.
"""
import importlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from testcontainers.community.ollama import OllamaContainer

# Small enough to pull and run on CPU in a reasonable time. It doesn't always
# follow the comma-separated output format perfectly, which is why the test
# below only checks that *something* came back rather than exact keywords.
MODEL_NAME = "qwen2.5:0.5b"

# Cache pulled models under the host's home directory (mapped into the
# container's /root/.ollama) so repeat test runs reuse them instead of
# re-downloading every time. Deliberately a dedicated directory rather than
# the host's real ~/.ollama, so this never shares state with (or gets
# corrupted by version differences from) an actual Ollama install on the host.
OLLAMA_MODEL_CACHE_DIR = Path(
    os.environ.get("OLLAMA_TEST_CACHE_DIR", Path.home() / ".cache" / "radar-keyword-search-ollama-test")
)


@pytest.fixture(scope="module")
def ollama_backend():
    # runIntegrationTestsDocker.sh starts its own Ollama server directly (as
    # a sibling container on the host, not via testcontainers) and pulls the
    # model into it *before* this test container even starts, then points
    # us at it via OLLAMA_BASE_URL. That sidesteps Docker-outside-of-Docker
    # entirely - no docker.sock mount, no separate proxy setup for a
    # container started through the raw Docker Engine API. When that's
    # already been done, just use it instead of starting a second server.
    if os.environ.get("OLLAMA_BASE_URL"):
        os.environ.setdefault("EXTRACTION_BACKEND", "ollama")
        os.environ.setdefault("OLLAMA_MODEL", MODEL_NAME)

        import iri_api

        importlib.reload(iri_api)
        yield iri_api
        return

    OLLAMA_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # OllamaContainer auto-requests a GPU device when the Docker daemon
    # reports an "nvidia" runtime. Docker Desktop's WSL2 backend can report
    # that runtime as available even without a real GPU attached, which
    # then fails at container start ("no adapters were found"). with_kwargs()
    # resets the container's docker-run kwargs, dropping that GPU request so
    # this always runs on CPU, matching what "CPU only" actually needs.
    container = OllamaContainer(ollama_home=OLLAMA_MODEL_CACHE_DIR).with_kwargs()

    # testcontainers talks to the Docker Engine API directly rather than
    # through the `docker` CLI, so it never picks up a proxy from the CLI's
    # own config the way our top-level `docker build`/`run` do. Behind a
    # corporate proxy, this container otherwise has no route to the
    # internet to actually pull the model. Forward whatever proxy this
    # process itself is using (baked in via Dockerfile.integration when run
    # through runIntegrationTestsDocker.sh) so `ollama pull` can reach it.
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        proxy_value = os.environ.get(proxy_var)
        if proxy_value:
            container = container.with_env(proxy_var, proxy_value)

    # Ollama's server only logs the outcome of an outbound registry request
    # (DNS/TLS/proxy CONNECT failures etc.) when this is enabled - without
    # it, a failed pull produces nothing but the generic "something went
    # wrong" client-side error with no way to tell why.
    container = container.with_env("OLLAMA_DEBUG", "1")

    with container as ollama:
        # ollama.pull_model() shells out to `self.exec(...)` and never checks
        # the exit code, so a failed pull (e.g. no network route to the
        # model registry) is silently ignored - the model would then just be
        # missing once the test actually tries to use it. Run the pull
        # ourselves and fail fast with the real error instead.
        pull_result = ollama.exec(f"ollama pull {MODEL_NAME}")
        if pull_result.exit_code != 0:
            stdout, stderr = ollama.get_logs()
            raise AssertionError(
                f"'ollama pull {MODEL_NAME}' failed with exit code {pull_result.exit_code}:\n"
                f"{pull_result.output.decode(errors='replace')}\n"
                f"--- ollama server stdout ---\n{stdout.decode(errors='replace')}\n"
                f"--- ollama server stderr ---\n{stderr.decode(errors='replace')}"
            )

        os.environ["EXTRACTION_BACKEND"] = "ollama"
        os.environ["OLLAMA_BASE_URL"] = f"{ollama.get_endpoint()}/v1"
        os.environ["OLLAMA_MODEL"] = MODEL_NAME

        import iri_api

        importlib.reload(iri_api)
        yield iri_api


def test_ollama_backend_extracts_keywords_from_a_real_server(ollama_backend, monkeypatch):
    # Keep this test focused on the Ollama wiring; don't depend on the real
    # TIB Terminology API being reachable/stable in CI.
    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return {
            "iri": f"https://example.org/{keyword}",
            "label": keyword,
            "best_term": keyword,
            "distance": 0,
            "ontology_name": "test",
        }

    monkeypatch.setattr(ollama_backend, "search_tib_best_match", fake_search)

    with TestClient(ollama_backend.app) as client:
        response = client.post(
            "/extract-iris",
            json={"document": "Insulin regulates blood glucose levels in patients with diabetes mellitus."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body, "expected the real Ollama model to return at least one keyword"
    # Every value should be our fake TIB match, proving the keywords that
    # came back from the real Ollama call actually made it through the
    # extraction -> TIB-lookup pipeline.
    assert all(match["ontology_name"] == "test" for match in body.values())
