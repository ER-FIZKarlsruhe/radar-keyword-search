
rd-search with TIB Terminology Service Support

This service provides keyword extraction from documents using one of two interchangeable backends — a custom PubMedBERT model (CPU only) or a local Ollama model (GPU-accelerated) — followed by entity linking to the [TIB Terminology Service](https://api.terminology.tib.eu).

---

## 🚀 Setup

### 1. Install Requirements

```bash
python -m radar-keywords-env  && source radar-keywords-env/bin/activate

pip install -r requirements.txt
```

### 2. Choose a Backend

The server picks **one** backend at startup, controlled by `EXTRACTION_BACKEND`. Only that
backend's dependencies are loaded/required — selecting `pubmedbert` never needs an Ollama
server, and selecting `ollama` never downloads the PubMedBERT model.

| `EXTRACTION_BACKEND` | Description | Extra env vars |
|---|---|---|
| `pubmedbert` (default) | Local PubMedBERT model, runs on CPU, no external services | — |
| `ollama` | A local/self-hosted [Ollama](https://ollama.com) model via its OpenAI-compatible API — use this on a machine with a dedicated GPU | `OLLAMA_BASE_URL` (optional, default `http://localhost:11434/v1`), `OLLAMA_MODEL` (optional, default `llama3`) |

```bash
# Example: run against a local Ollama server with a specific model
export EXTRACTION_BACKEND=ollama
export OLLAMA_MODEL=mistral
```

> The active backend is reported by the `/` health check endpoint.

---

## 🧠 Models Used

* **`pubmedbert`**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`), CPU only
* **`ollama`**: Any chat-capable model served by a local Ollama instance, via KeyBERT's `KeyLLM` interface

---

## ▶️ Run the Server

Use `uvicorn` to launch the FastAPI app on port `8001` (or any other port of your choice):

```bash
uvicorn iri_api:app --reload --port 8001
```

---

## 🧪 Usage Example

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

### Ollama Integration Test

`integration_tests/` additionally has a real end-to-end test of the Ollama backend: it starts an
actual Ollama server in a Docker container ([testcontainers](https://testcontainers.com)), pulls a
small model (`qwen2.5:0.5b`) into it, and sends a real request through the FastAPI app. Unlike the
suite above, this uses the **real** `keybert`/`openai`/`torch` packages (not stand-ins) and needs
**Docker** running locally. It's deliberately kept separate from — and out of the default `pytest`
run for — the fast suite above.

```bash
pip install -r integration_tests/requirements.txt
pytest integration_tests
```

The pulled model is cached under `~/.cache/radar-keyword-search-ollama-test` on the host (mapped
into the container), so repeat runs reuse it instead of re-downloading every time. Override the
location with the `OLLAMA_TEST_CACHE_DIR` environment variable if needed.

> The container always runs on CPU, even on a machine with an NVIDIA GPU — this test only needs to
> prove the Ollama *wiring* works, not benchmark inference speed.

---

## Linux Service
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
