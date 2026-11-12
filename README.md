# radar-keyword-search with TIB Terminology Service Support

This service extracts keywords from documents with a local BERT model (PubMedBERT by default, CPU only) and links them to the [TIB Terminology Service](https://api.terminology.tib.eu), the
[GND](https://sta.dnb.de/doc/GND) (Gemeinsame Normdatei, via [lobid-gnd](https://lobid.org/gnd)), or both.

> **How to run this service:** in **production, always use the Docker image** (`docker/dockerfile`) — see [Running in Production](#-running-in-production-docker) below. The `python`/`pip` commands further down are for **local development only** (running tests, debugging, iterating on `iri_api.py`).

---

## 🚀 Running in Production (Docker)

### 1. Start the Container

```bash
docker run -d --name radar-keyword-search \
  -p 8000:8000 \
  docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1
```

This needs no external services besides the terminology APIs. Startup is slow while the default
BERT model (PubMedBERT) downloads/loads — check
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

Link the keywords to GND subject headings instead of TIB (`"both"` searches TIB and GND and keeps
the better match per keyword):

```bash
curl -v --noproxy '*' -X POST http://localhost:8000/extract-iris   -H 'Content-Type: application/json'   -d '{"document": "Historisch-kritische Edition mit Schauspielmusik und Forschungsdatenmanagement", "model": "gbert", "terminology": "gnd"}'
```

Same, but restricted to a specific ontology (`ncit`):

```bash
curl -v --noproxy '*' -X POST http://localhost:8000/extract-iris \
  -H 'Content-Type: application/json' \
  -d '{"document": "The mitochondria is the powerhouse of the cell and regulates apoptosis.", "ontology": "ncit"}'
```

Rank noun phrases found by spaCy instead of single words, so multi-word terms like
`historisch-kritische Edition` stay whole (`language` is optional, see
[How It Works](#-how-it-works)):

```bash
curl -v --noproxy '*' -X POST http://localhost:8000/extract-iris \
  -H 'Content-Type: application/json' \
  -d '{"document": "Historisch-kritische Edition mit Schauspielmusik und Forschungsdatenmanagement", "model": "gbert", "terminology": "gnd", "candidates": "phrases", "language": "de"}'
```

> `--noproxy '*'` avoids routing the request to `localhost` through a corporate proxy set via
> `HTTP_PROXY`/`HTTPS_PROXY` on your machine — omit it if you don't have one configured.

### 3. Choosing the BERT Model

Each request can pick a model with the `model` field; `GET /models` lists the available ones (see
`bert_models.py`). The model used when a request names none is set at container start:

```bash
docker run -d --name radar-keyword-search \
  -p 8000:8000 \
  -e BERT_MODEL=gbert \
  docker.dev.fiz-karlsruhe.de/radar-keyword-search:0.1
```

* `BERT_MODEL`: optional, defaults to `pubmedbert`. The default model is loaded at startup, any
  other model on its first request (and then cached).

The default model is reported by the `/` health check endpoint. `docker/docker-compose.prod.yml`
runs the same container via `docker compose -f docker/docker-compose.prod.yml up -d`.

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

### 2. Run the Server

Use `uvicorn` to launch the FastAPI app on port `8001` (or any other port of your choice):

```bash
uvicorn iri_api:app --reload --port 8001
```

### 3. Usage Example

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

* Default: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`), CPU only
* Alternatives (SciBERT, BioBERT, GBERT, multilingual E5, ...): see `bert_models.py` or `GET /models`

---

## 🔍 How It Works

1. **Keyword Extraction** (`/extract-iris`):

   The [KeyBERT](https://github.com/MaartenGr/KeyBERT) algorithm by Maarten Grootendorst, with
   BERT embeddings: the document's own words (minus English and German stop words, see
   `stop_words.py`) are ranked by cosine
   similarity to the whole document. It lives in `keyword_extraction.py`, a small port of the
   code path of KeyBERT 0.9.0 that this service uses, with identical results; the `keybert`
   package itself is no longer a dependency. See [Acknowledgements](#-acknowledgements).

   What is ranked is chosen per request via `candidates`:

   * `words` (default): the document's single words, as described above.
   * `ngrams`: its sequences of up to three words (after stop-word removal, so also
     sequences that are no phrase, e.g. `regulates apoptosis`).
   * `phrases`: its noun phrases and nouns, found by [spaCy](https://spacy.io)
     (`noun_phrases.py`, pipelines `de_core_news_md` / `en_core_web_md`). Multi-word terms
     like `type 2 diabetes mellitus` or `historisch-kritische edition` stay whole. Each
     paragraph is parsed as German or English, whichever language has more stop words in it
     (a paragraph without any, e.g. a title, takes the whole document's language);
     `language` (`de`/`en`) overrides that. Articles and other function words are stripped
     from a phrase's edges, and phrases longer than four words are cut to their last four
     (the head noun and its closest modifiers). The spaCy pipelines are loaded on the first
     request that asks for `phrases`.

2. **IRI Linking** (chosen per request via `terminology`):

   * `tib` (default): each keyword is searched in the TIB Terminology API (optionally scoped by
     `ontology` / `ontology_collection`). The closest label/synonym by Hamming distance wins; it is
     `matched` if the distance is within the allowed distance (see `max_distance` below) and its IRI isn't provably dead:
     only a HEAD answer of 404/410, or a host that can't be resolved or refuses the connection,
     drops the match. Other statuses and timeouts keep it, since hosts behind bot protection
     (e.g. `www.loc.gov` for PREMIS terms) answer automated requests with 403, a 503 challenge
     or by stalling. Every non-200 answer is logged.
   * `gnd`: each keyword is searched in the GND via lobid-gnd, restricted to the types in
     `GND_TYPES`. The closest preferred/variant name wins (on a tie, the type listed first in
     `GND_TYPES`); it is `matched` if the distance is within the allowed distance (no HEAD check - lobid only
     returns live records). `ontology` and `ontology_collection` are ignored.
     * Persons are only matched for keywords of at least two words: a single word like `goethe` or
       `napoleon` would otherwise hit some person whose (variant) name is exactly that. Inverted
       names are also compared in natural order, so `Johann Wolfgang von Goethe` matches the GND's
       `Goethe, Johann Wolfgang von`. With the default `candidates: words` the extractor only
       produces single words, so person matches only occur with `ngrams` or `phrases`.
   * `both`: both lookups run in parallel; a matched candidate beats an unmatched one, then the
     smaller distance wins, and on a tie TIB wins.
   * Allowed distance: a quarter of the keyword's length (rounded down), and at most
     `max_distance` (default 5). Keywords under 4 letters need an exact match, 4–7 letters may
     differ by 1, 8–11 by 2, and so on. This accepts small edits such as `nanoparticles` →
     `nanoparticle` or `gluck` → `Glück`. It rejects what a fixed threshold of 5 let through:
     any term for a short keyword (`hier` → `usko`) and a keyword plus extra words (`witches` →
     `Witches game`; the Hamming distance pads the shorter string, so such a prefix match only
     costs the extra characters). `"max_distance": 0` accepts exact matches only (ignoring case).
     The closest candidate is always returned, with `matched: false` when it is too far.
   * The GND is German: English keywords from English-only models (e.g. `music`) rarely hit a GND
     heading (`Musik`) exactly. Prefer a German or multilingual model (`gbert`,
     `multilingual-e5`, ...) for German documents linked to the GND.

---

## 🛠 Configuration

* **TIB API Endpoint**: `https://api.terminology.tib.eu/api/search`
* **GND API Endpoint**: `https://lobid.org/gnd/search`
* **Environment Variables**:

  * `GND_SEARCH_URL`: Optional, defaults to `https://lobid.org/gnd/search`
  * `GND_TYPES`: Optional, comma-separated GND entity types searched for `gnd`/`both`, in order
    of preference, defaults to `SubjectHeading,PlaceOrGeographicName,CorporateBody,Person`. Works
    and conferences are deliberately left out: they dominate unfiltered results (e.g. `Goethe`
    returns five *Works* first).
  * `BERT_MODEL`: Optional, defaults to `pubmedbert`; the model used when a request names none
  * `MAX_DOCUMENT_CHARS`: Optional, defaults to `100000`; longer documents are rejected with
    HTTP 422. This bounds the work one request can cause: every distinct word is a candidate
    keyword, and a 100,000-character document of distinct words takes about 50 s of CPU.
    Keyword extraction runs one request at a time in a worker thread, so a long extraction
    delays other extractions but not health checks or other requests' TIB/GND lookups.

---

## 📄 Input Schema

```json
{
  "document": "Text to extract keywords from, at most MAX_DOCUMENT_CHARS (default 100000) characters",
  "terminology": "Optional: tib (default) | gnd | both",
  "ontology": "Optional TIB ontology id(s) for filtering (e.g., et); TIB only",
  "ontology_collection": "Optional TIB collection id(s) (e.g., NFDI4CHEM); TIB only",
  "model": "Optional BERT model name, see GET /models",
  "max_keywords": "Optional, 1-20, default 10",
  "max_distance": "Optional, 0-5, default 5; 0 = exact matches only; shorter keywords get less (a quarter of their length)",
  "candidates": "Optional: words (default) | ngrams | phrases; what is ranked as keywords, see How It Works",
  "language": "Optional: de | en; the document's language for candidates=phrases (default: detected per paragraph)"
}
```

---

## 🗞 Response Format

```json
{
  "insulin": {
    "iri": "http://purl.obolibrary.org/obo/CHEBI_5931",
    "label": "insulin",
    "best_term": "insulin",
    "distance": 0,
    "matched": true,
    "ontology_name": "chebi",
    "source": "tib",
    "keyword_scheme": "Other",
    "scheme_uri": "http://purl.obolibrary.org/obo/",
    "classification_code": "CHEBI_5931"
  },
  "Schauspielmusik": {
    "iri": "https://d-nb.info/gnd/4280160-6",
    "label": "Schauspielmusik",
    "best_term": "Schauspielmusik",
    "distance": 0,
    "matched": true,
    "ontology_name": null,
    "source": "gnd",
    "keyword_scheme": "GND",
    "scheme_uri": "https://d-nb.info/gnd/",
    "classification_code": "4280160-6"
  },
  "keyword3": null
}
```

* `matched`: whether the candidate is good enough to link (distance within the allowed distance -
  a quarter of the keyword's length, at most `max_distance` - and for TIB a live IRI).
  The closest candidate is returned even when it isn't, so clients can show a match-quality rating.
* `source`: `tib` or `gnd` - which terminology the candidate came from.
* `keyword_scheme`, `scheme_uri`, `classification_code`: ready-made values for a RADAR keyword's
  `keywordScheme` (`GND` / `Other`), `schemeURI` and `classificationCode` attributes.
* `label`: the term's preferred label; `best_term`: the label or synonym/variant that actually matched.

> The value is `null` only if the search failed or returned no candidates at all.

---


## 🧪 Running the Tests

The test suite does **not** need the ML stack (torch, transformers) installed, model
downloads, or network access — `tests/conftest.py`
replaces those heavy dependencies with lightweight stand-ins before `iri_api` is imported,
and configures each one per test. Only the small web-framework packages plus numpy and
scikit-learn (for the real keyword-ranking tests in `tests/test_keyword_extraction.py`) are
needed to run it:

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
The integration tests below have an equivalent `bin/runIntegrationTestsDocker.sh`.

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

This covers the keyword ranking (`keyword_extraction.py`), `hamming_distance`, `check_iri_exists`,
`search_tib_best_match`, the GND lookup, and `/extract-iris` (success, error, and edge cases).

### Integration Tests

`integration_tests/` additionally has real end-to-end tests that talk to actual external services
instead of mocking them. Like the fast suite, they're run with `pytest`, but they're deliberately
kept separate from — and out of the default `pytest` run for (see `testpaths` in `pytest.ini`) —
the suite above.

```bash
pip install -r integration_tests/requirements.txt
pytest integration_tests
```

#### GND (lobid) Test

`test_gnd_service.py` does the same against the real `https://lobid.org/gnd` API, using GND
subject headings from a real RADAR4Culture dataset, and checks the `both` mode against TIB + lobid.
It needs outbound network access to `lobid.org` and `api.terminology.tib.eu`.

#### TIB Terminology Service Test

`test_tib_service.py` sends a real request through the FastAPI app against the actual
`https://api.terminology.tib.eu` API (no mocking), and does a real `HEAD` request against the IRI
it resolves — proving the TIB lookup pipeline (`search_tib_best_match` / `check_iri_exists`) works
end-to-end. It needs outbound network access to `api.terminology.tib.eu` and to whatever ontology
registry hosts the resolved IRIs (e.g. `purl.obolibrary.org` for ChEBI). Keyword extraction is
faked with a fixed keyword list (and the ML packages are stubbed, like in the fast suite), so this
test is only about the TIB integration, not keyword extraction.

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
Environment="BERT_MODEL=pubmedbert"
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

---

## 🙏 Acknowledgements

The keyword extraction is based on **[KeyBERT](https://github.com/MaartenGr/KeyBERT)** by
Maarten Grootendorst. This service used the `keybert` package until it was replaced by
`keyword_extraction.py`, a port of the code path we use (from KeyBERT 0.9.0). The algorithm and
its design are KeyBERT's; thank you for making it available.

KeyBERT is released under the MIT License, which allows its code to be used, modified and
redistributed within this Apache-2.0 project, provided its copyright and permission notice are
kept. That notice is in the header of `keyword_extraction.py` and in [NOTICE](NOTICE), which is
also shipped in the Docker image.

The German stop words in `stop_words.py` are the [Snowball](https://snowballstem.org) project's
German stop word list (Copyright (c) 2001 Dr Martin Porter, 2002 Richard Boulton), used under the
3-clause BSD License; see [NOTICE](NOTICE).

Noun phrases (`candidates: phrases`) are found with [spaCy](https://spacy.io) and its pipelines
`de_core_news_md` and `en_core_web_md` by Explosion AI (MIT License; their training data is
credited in [NOTICE](NOTICE)).

If you use this service in research, please cite KeyBERT:

```bibtex
@misc{grootendorst2020keybert,
  author       = {Maarten Grootendorst},
  title        = {KeyBERT: Minimal keyword extraction with BERT.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.4461265},
  url          = {https://doi.org/10.5281/zenodo.4461265}
}
```

---

## ⚖️ License

Copyright 2025-2026 FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur GmbH

Licensed under the [Apache License, Version 2.0](LICENSE). You may not use this project except in compliance with the License. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](LICENSE) file for the specific language governing permissions and limitations.

`keyword_extraction.py` contains code derived from [KeyBERT](https://github.com/MaartenGr/KeyBERT),
Copyright (c) 2020 Maarten P. Grootendorst, used under the MIT License; see [NOTICE](NOTICE).
