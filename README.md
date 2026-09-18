
rd-search with TIB Terminology Service Support

This service provides keyword extraction from documents using either a custom PubMedBERT-based model or OpenAI's ChatGPT (via KeyBERT LLM), followed by entity linking to the [TIB Terminology Service](https://api.terminology.tib.eu).

---

## 🚀 Setup

### 1. Install Requirements

```bash
python -m radar-keywords-env  && source radar-keywords-env/bin/activate

pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export CHAT_GPT_API_KEY=your_openai_api_key_here
```

> This is required for the `/extract-iris-openai` endpoint. If the variable is not set, the server will raise a runtime error on startup.

---

## 🧠 Models Used

* **Custom Model**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
* **LLM (Optional)**: ChatGPT via KeyBERT's `KeyLLM` interface

---

## ▶️ Run the Server

Use `uvicorn` to launch the FastAPI app on port `8001` (or any other port of your choice):

```bash
uvicorn iri_api:app --reload --port 8001
```

---

## 🧪 Usage Examples

### 1. Extract IRIs using OpenAI (ChatGPT)

```bash
curl --noproxy '*' -X POST http://localhost:8001/extract-iris-openai \
  -H "Content-Type: application/json" \
  -d '{
        "document": "Experimental Data to the publication \"Mononuclear and multinuclear O^N^O-donor Zn(II) complexes as robust catalysts for the production and depolymerization of poly(lactide)\"",
        "ontology": "et"
      }'
```

### 2. Extract IRIs using PubMedBERT

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

1. **Keyword Extraction**:

   * `/extract-iris`: Uses `KeyBERT` with PubMedBERT
   * `/extract-iris-openai`: Uses `KeyLLM` with ChatGPT

2. **IRI Linking via TIB**:

   * For each extracted keyword, a search is sent to the TIB Terminology API.
   * The best match is selected based on the Hamming distance threshold (`<= 5`) and IRI availability.
   * Only valid IRIs (HTTP 200 on HEAD request) are returned.

---

## 🛠 Configuration

* **TIB API Endpoint**: `https://api.terminology.tib.eu/api/search`
* **Environment Variable**:

  * `CHAT_GPT_API_KEY`: Required for OpenAI keyword extraction

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
installed, a GPU, network access, or a real `CHAT_GPT_API_KEY` — `tests/conftest.py`
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

This covers `hamming_distance`, `check_iri_exists`, `search_tib_best_match`, and both
`/extract-iris` endpoints (success, error, and edge cases like blank keywords from the LLM).

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
Environment="CHAT_GPT_API_KEY=your_value_here"
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
