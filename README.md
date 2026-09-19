
rd-search with TIB Terminology Service Support

This service provides keyword extraction from documents using one of two interchangeable backends — a custom PubMedBERT model (CPU only) or a local Ollama model — followed by entity linking to the [TIB Terminology Service](https://api.terminology.tib.eu).

> **How to run this service:** in **production, always use the Docker image** (`docker/dockerfile`) — see [Running in Production](#-running-in-production-docker) below. The `python`/`pip` commands further down are for **local development only** (running tests, debugging, iterating on `iri_api.py`).

---

## 🚀 Running in Production (Docker)

### 1. Start the Container

```bash
docker run -d --name radar-keyword-search \
  -p 8000:8000 \
  docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1
```

This starts with the default backend (`pubmedbert`, CPU only, no external services needed). The
first request after startup will be slow while the PubMedBERT model downloads/loads — check
`docker logs radar-keyword-search` and wait for `Uvicorn running on http://0.0.0.0:8000` before
sending requests.

### 2. Test It

Health check:

```bash
curl -v --noproxy '*' http://localhost:8000/
```

Extract keywords/IRIs from a document:

```bash
curl -v --noproxy '*' -X POST http://localhost:8000/extract-iris \
  -H 'Content-Type: application/json' \
  -d '{"document": "The mitochondria is the powerhouse of the cell and regulates apoptosis."}'
```

Same, but restricted to a specific ontology (`ncit`):

```bash
curl -v --noproxy '*' -X POST http://localhost:8000/extract-iris \
  -H 'Content-Type: application/json' \
  -d '{"document": "The mitochondria is the powerhouse of the cell and regulates apoptosis.", "ontology": "ncit"}'
```

> `--noproxy '*'` avoids routing the request to `localhost` through a corporate proxy set via
> `HTTP_PROXY`/`HTTPS_PROXY` on your machine — omit it if you don't have one configured.

### 3. Switching Backend / Ollama Model in Production

The backend is picked at container start via the `EXTRACTION_BACKEND` environment variable — pass
it (and any backend-specific variables) with `-e` on `docker run`.

First, start an Ollama server and pull a model into it. `llama3` is the default — it follows the
comma-separated keyword-list format KeyBERT's `KeyLLM` prompts for reliably; much smaller models
(e.g. `qwen2.5:0.5b`) tend to ignore that formatting instruction and return everything as a single
garbled keyword instead of a clean list. Put Ollama on its own Docker network so the
`radar-keyword-search` container can reach it by name:

```bash
docker network create radar-net

docker run -d --name ollama \
  --network radar-net \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama:latest

docker exec ollama ollama pull llama3
```

`llama3` is a ~4.7GB download, so the pull takes a few minutes. `-p 11434:11434` is only needed if
you also want to reach Ollama directly from the Docker host (e.g. `curl localhost:11434`); the
`radar-keyword-search` container reaches it via the shared network regardless. `-v ollama:/root/.ollama`
persists pulled models across container restarts.

Then start the service pointed at it:

```bash
docker run -d --name radar-keyword-search \
  --network radar-net \
  -p 8000:8000 \
  -e EXTRACTION_BACKEND=ollama \
  -e OLLAMA_BASE_URL=http://ollama:11434/v1 \
  -e OLLAMA_MODEL=llama3 \
  docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1
```

* `EXTRACTION_BACKEND`: `pubmedbert` (default) or `ollama`
* `OLLAMA_BASE_URL`: where the Ollama server's OpenAI-compatible API is reachable **from inside the
  container** — `localhost` here means the container itself, not the Docker host, so this must be
  an address the container can actually resolve/reach:
  * Ollama running in another container on the same Docker network: use that container's name,
    e.g. `http://ollama:11434/v1`, and start both containers with `--network <shared-network>`.
  * Ollama running directly on the Docker host: use `http://host.docker.internal:11434/v1`
    (Docker Desktop) or the host's LAN/bridge IP on Linux (add `--add-host=host.docker.internal:host-gateway`
    to `docker run` to get the same hostname to work there too).
* `OLLAMA_MODEL`: any chat-capable model already pulled into that Ollama server, e.g. `llama3`
  (default), `mistral`. Smaller models like `qwen2.5:0.5b` are useful for a quick wiring smoke
  test, but not recommended for real keyword-extraction quality — see the note above.
* `OLLAMA_HTTP_PROXY`: optional — only needed if the Ollama server is itself behind a proxy (by
  default, calls to Ollama bypass any proxy set via `HTTP_PROXY`/`HTTPS_PROXY`, since Ollama is
  typically local/internal and a corporate proxy would otherwise intercept and break those calls).

The active backend is reported by the `/` health check endpoint.

### 3a. Reusing an Existing Ollama Server (docker-compose)

If Ollama is already running on the host as its own container (e.g. as part of an Open WebUI
stack), don't start a second one — join the existing Docker network instead and let
`radar-keyword-search` reach it by container name. `docker/docker-compose.prod.yml` does this:

```bash
docker compose -f docker/docker-compose.prod.yml up -d
```

Edit the `OLLAMA_BASE_URL`/`OLLAMA_MODEL` environment values and the `networks.ollama-net.name` in
that file to match your host's actual Ollama container name/network — see the comment at the top
of the file for how to find it.

### 4. Building the Image Yourself

The image is built from `docker/dockerfile` (multi-stage: CPU-only PyTorch wheels, no CUDA stack,
no compiler toolchain in the final image):

```bash
docker build -f docker/dockerfile -t radar-keyword-search:local .
```

On a build agent behind a corporate proxy (e.g. Bamboo), pass it through as build args:

```bash
docker build -f docker/dockerfile \
  --build-arg HTTP_PROXY="$HTTP_PROXY" \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  --build-arg NO_PROXY="$NO_PROXY" \
  -t radar-keyword-search:local .
```

---

## 🧑‍💻 Local Development Setup (Python / pip)

> This is for developing/debugging the service locally — production deployments use the Docker
> image above, not these commands.

### 1. Install Requirements

```bash
python -m venv radar-keywords-env && source radar-keywords-env/bin/activate

pip install -r requirements.txt
```

### 2. Choose a Backend

The server picks **one** backend at startup, controlled by `EXTRACTION_BACKEND`. Only that
backend's dependencies are loaded/required — selecting `pubmedbert` never needs an Ollama
server, and selecting `ollama` never downloads the PubMedBERT model.

| `EXTRACTION_BACKEND` | Description | Extra env vars |
|---|---|---|
| `pubmedbert` (default) | Local PubMedBERT model, runs on CPU, no external services | — |
| `ollama` | A local/self-hosted [Ollama](https://ollama.com) model via its OpenAI-compatible API | `OLLAMA_BASE_URL` (optional, default `http://localhost:11434/v1`), `OLLAMA_MODEL` (optional, default `llama3`) |

```bash
# Example: run against a local Ollama server with a specific model
export EXTRACTION_BACKEND=ollama
export OLLAMA_MODEL=mistral
```

> The active backend is reported by the `/` health check endpoint.

### 3. Run the Server

Use `uvicorn` to launch the FastAPI app on port `8001` (or any other port of your choice):

```bash
uvicorn iri_api:app --reload --port 8001
```

### 4. Usage Example

There is a single endpoint regardless of backend — whichever one is active via `EXTRACTION_BACKEND`
is used automatically:

```bash
curl --noproxy '*' -X POST http://localhost:8001/extract-iris \
  -H "Content-Type: application/json" \
  -d '{
        "document": "Experimental Data to the publication \"Mononuclear and multinuclear O^N^O-donor Zn(II) complexes as robust catalysts for the production and depolymerization of poly(lactide)\"",
        "ontology": "et"
      }'
```

---

## 🧠 Models Used

* **`pubmedbert`**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`), CPU only
* **`ollama`**: Any chat-capable model served by a local Ollama instance, via KeyBERT's `KeyLLM` interface

---

## 🔍 How It Works

1. **Keyword Extraction** (`/extract-iris`):

   * `pubmedbert` backend: uses `KeyBERT` with PubMedBERT
   * `ollama` backend: uses `KeyLLM`, pointed at a local Ollama server's OpenAI-compatible chat API

2. **IRI Linking via TIB**:

   * For each extracted keyword, a search is sent to the TIB Terminology API.
   * The best match is selected based on the Hamming distance threshold (`<= 5`) and IRI availability.
   * Only valid IRIs (HTTP 200 on HEAD request) are returned.

---

## 🛠 Configuration

* **TIB API Endpoint**: `https://api.terminology.tib.eu/api/search`
* **Environment Variables**:

  * `EXTRACTION_BACKEND`: `pubmedbert` (default) or `ollama`
  * `OLLAMA_BASE_URL`: Optional, defaults to `http://localhost:11434/v1`
  * `OLLAMA_MODEL`: Optional, defaults to `llama3`
  * `OLLAMA_HTTP_PROXY`: Optional. By default, calls to Ollama bypass any system-configured
    proxy entirely (Ollama is typically local/internal, and a corporate proxy can otherwise
    intercept and break those calls). Set this if Ollama itself is only reachable through a proxy.

---

## 📄 Input Schema

```json
{
  "document": "Text to extract keywords from",
  "ontology": "Optional ontology identifier for filtering (e.g., et)"
}
```

---

## 🗞 Response Format

```json
{
  "keyword1": {
    "iri": "...",
    "label": "...",
    "best_term": "...",
    "distance": 2,
    "ontology_name": "..."
  },
  "keyword2": null
}
```

> If no match is found under the Hamming threshold or IRI validation fails, the value will be `null`.

---


## 🧪 Running the Tests

The test suite does **not** need the GPU/ML stack (torch, transformers, keybert, openai)
installed, a GPU, network access, or a running Ollama server — `tests/conftest.py`
replaces those heavy dependencies with lightweight stand-ins before `iri_api` is imported,
and configures each one per test. Only the small web-framework packages are needed to run it:

**Linux / macOS:**
```bash
python -m venv radar-keywords-test-env && source radar-keywords-test-env/bin/activate

pip install -r requirements-test.txt

pytest -q --cov=iri_api --cov-report=term-missing
```

This exact flow is also checked in as `bin/runTests.sh` (used by our Bamboo CI plan), which
additionally honors `HTTP_PROXY`/`HTTPS_PROXY` when installing dependencies — needed on CI agents
that have no direct internet access and can only reach PyPI through a corporate proxy:

```bash
HTTP_PROXY=http://proxy.example.com:8080 ./bin/runTests.sh
```

Bamboo actually runs the tests in a container instead, via `bin/runTestsDocker.sh` (which builds
`docker/Dockerfile.test`) — this avoids depending on whatever Python happens to be installed on the
build agent. It writes a JUnit XML report to `test-results/junit.xml` for Bamboo's test results.
The Ollama integration test below has an equivalent `bin/runIntegrationTestsDocker.sh`.

Leave `HTTP_PROXY`/`HTTPS_PROXY` unset for a normal, direct install (e.g. on a developer machine
with unrestricted internet access).

**Windows (PowerShell):**
```powershell
python -m venv radar-keywords-test-env

.\radar-keywords-test-env\Scripts\Activate.ps1

pip install -r requirements-test.txt

pytest -q --cov=iri_api --cov-report=term-missing
```

> If `python -m venv` fails with an `ensurepip` error, you are likely running it from a
> Cygwin/MSYS shell whose `python` is a Unix-style build with a broken pip bootstrap.
> Run the commands above from PowerShell or Command Prompt instead (using Windows Python,
> e.g. Anaconda's), or Git Bash.

This covers `hamming_distance`, `check_iri_exists`, `search_tib_best_match`, and `/extract-iris`
under both backends (success, error, and edge cases like blank keywords from the LLM).

### Integration Tests

`integration_tests/` additionally has real end-to-end tests that talk to actual external services
instead of mocking them. Like the fast suite, they're run with `pytest`, but they're deliberately
kept separate from — and out of the default `pytest` run for (see `testpaths` in `pytest.ini`) —
the suite above.

```bash
pip install -r integration_tests/requirements.txt
pytest integration_tests
```

#### Ollama Backend Test

`test_ollama_backend.py` starts an actual Ollama server in a Docker container
([testcontainers](https://testcontainers.com)), pulls a small model (`qwen2.5:0.5b`) into it, and
sends a real request through the FastAPI app. Unlike the fast suite, this uses the **real**
`keybert`/`openai`/`torch` packages (not stand-ins) and needs **Docker** running locally. The TIB
Terminology API call is faked, so this test is only about the Ollama wiring.

The pulled model is cached under `~/.cache/radar-keyword-search-ollama-test` on the host (mapped
into the container), so repeat runs reuse it instead of re-downloading every time. Override the
location with the `OLLAMA_TEST_CACHE_DIR` environment variable if needed.

> The container always runs on CPU, even on a machine with an NVIDIA GPU — this test only needs to
> prove the Ollama *wiring* works, not benchmark inference speed.

#### TIB Terminology Service Test

`test_tib_service.py` sends a real request through the FastAPI app against the actual
`https://api.terminology.tib.eu` API (no mocking), and does a real `HEAD` request against the IRI
it resolves — proving the TIB lookup pipeline (`search_tib_best_match` / `check_iri_exists`) works
end-to-end. It needs outbound network access to `api.terminology.tib.eu` and to whatever ontology
registry hosts the resolved IRIs (e.g. `purl.obolibrary.org` for ChEBI). Keyword extraction is
faked with a fixed keyword list (and the ML packages are stubbed, like in the fast suite), so this
test is only about the TIB integration, not any extraction backend.

---

## Alternative: Running as a systemd Service (non-Docker deployment)

> The Docker image is the recommended way to run this service in production (see
> [Running in Production](#-running-in-production-docker)). Use this instead only on a host where
> Docker isn't available and the service must run directly via the Python venv from
> [Local Development Setup](#-local-development-setup-python--pip).

### Create a systemd Service File

sudo nano /etc/systemd/system/radar-keywords.service

Paste this content (adjust paths to your setup):
```
[Unit]
Description=Radar Keywords API Service
After=network.target

[Service]
User=admin
Group=admin
WorkingDirectory=/data/radar-keyword-search
Environment="EXTRACTION_BACKEND=pubmedbert"
# On a GPU server running Ollama, use instead:
# Environment="EXTRACTION_BACKEND=ollama"
# Environment="OLLAMA_BASE_URL=http://localhost:11434/v1"
# Environment="OLLAMA_MODEL=llama3"
Environment="PATH=/data/radar-keyword-search/radar-keywords-env/bin"

ExecStart=/data/radar-keyword-search/radar-keywords-env/bin/python -m uvicorn iri_api:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target

```

### Reload and Enable the Service

* sudo systemctl daemon-reload

* sudo systemctl enable radar-keywords

* sudo systemctl start radar-keywords
