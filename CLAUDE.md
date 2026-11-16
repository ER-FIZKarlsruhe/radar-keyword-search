# CLAUDE.md

Notes for coding agents (Claude Code, Codex, ...) working on this repository.
User-facing documentation lives in `README.md` (and the OpenAPI docs at `/docs`); this file only holds what
an agent needs to work here efficiently and safely.

This repository is public. Never write internal hostnames, proxy addresses, registry URLs,
credentials or other internal details into files of this repository (including this one,
tests and examples). Refer to environment variables such as `$HTTP_PROXY` instead.

## What the service does

FastAPI service (`iri_api.py`) for RADAR: `POST /extract-iris` extracts keywords from a
document and optionally links them to TIB Terminology Service and/or GND (lobid-gnd).

| File | Role |
|------|------|
| `iri_api.py` | API, BERT model cache, TIB/GND lookup, matching rules (`hamming_distance`, `allowed_distance`) |
| `keyword_extraction.py` | KeyBERT algorithm ported from keybert 0.9.0 (cosine ranking against the document embedding) |
| `bert_models.py` | Registry of selectable BERT models (`GET /models`) |
| `pos_ngrams.py` | Phrase keywords of up to 3 words (`candidates: pos-ngrams`): spaCy POS tags for candidates, tuned re-ranking of the BERT similarities |
| `stop_words.py` | German + English stop words |
| `evaluation/` | Scripts that harvest RADAR metadata and evaluate the deployed service; `evaluation/benchmark/` scores the extraction alone |
| `evaluation/data/` | Git-ignored test data: RADAR harvest, benchmark sets and raw downloads, ChatGPT answers. Never commit or copy it into tracked files (public repo; KPBiomed is CC BY-NC) |
| `NOTICE` | Third-party attributions; update it whenever a dependency or data set with attribution duties is added |

## Conventions

- Code comments, docstrings and generated docs in English.
- Keep the existing style: explanatory comments that say *why*, small functions, plain dicts
  in responses.
- Tests are pytest (`tests/test_*.py`), async tests via `pytest-asyncio` (`asyncio_mode = auto`).
  HTTP calls are faked with `httpx.MockTransport` or by monkeypatching the lookup functions.
- When behaviour changes, update together: the field comments of `DocumentRequest` in `iri_api.py`,
  `README.md`, and labels in `evaluation/report.py` if they mention it.
- Do not commit, stage or create branches unless explicitly asked.

## Running the unit tests

The host's default `python3` is 3.6 and has no pytest; do not use it. Use Python 3.12 in a
virtualenv outside the repository (e.g. a scratch directory):

```bash
python3.12 -m venv <scratch>/venv
<scratch>/venv/bin/pip install -r requirements-test.txt
<scratch>/venv/bin/python -m pytest -q          # run from the repository root
```

`tests/conftest.py` replaces torch, transformers, `keyword_extraction` and `pos_ngrams`
with mocks, so no model download or network access is needed. spaCy and its two models are
real (installed from `requirements-test.txt`); `tests/test_pos_ngrams.py` loads the real
`pos_ngrams.py` from its file.

Alternative: `bin/runTestsDocker.sh` (builds `docker/Dockerfile.test`, as CI does).

## Network and proxy

- The development host reaches the internet only through the corporate proxy. The proxy is
  configured in the shell environment (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` and lowercase
  variants); `pip install` and other tools work because they pick these up.
- `docker build` does NOT inherit them: pass `--build-arg HTTP_PROXY="$HTTP_PROXY"
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" --build-arg NO_PROXY="$NO_PROXY"`
  (see `bin/runTestsDocker.sh`). The runtime stage of `docker/dockerfile` deliberately drops
  the proxy env.
- On the development host, build containers on Docker's default bridge network have no name
  resolution for the proxy host: the build hangs and pip finally fails with "No matching
  distribution found" (after several minutes). Build with `--network host` in addition to the
  proxy build args:

  ```bash
  docker build --network host \
    --build-arg HTTP_PROXY="$HTTP_PROXY" --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
    --build-arg NO_PROXY="$NO_PROXY" -f docker/Dockerfile.test -t radar-keyword-search-tests .
  ```

  `bin/runTestsDocker.sh` and `bin/runIntegrationTestsDocker.sh` don't pass `--network host`
  (the CI agents don't need it), so on the development host run the build by hand as above,
  or use the local venv instead (see "Running the unit tests"), which is faster anyway.
- If a container started with `docker run` (e.g. to install something for a one-off test)
  can't download, try `--network host` plus `-e HTTPS_PROXY="$HTTPS_PROXY"` etc. as well
  (only verified for `docker build`).
- A running service container needs the proxy at runtime for Hugging Face model downloads and
  the TIB/GND lookups; check with
  `docker inspect <container> --format '{{range .Config.Env}}{{println .}}{{end}}' | cut -d= -f1`
  (prints names only, no values).
- Hugging Face downloads on the host: the default Xet transfer backend (`hf-xet`) fails behind
  the proxy ("CAS Client Error: Request middleware error"). Set `HF_HUB_DISABLE_XET=1` to
  use plain HTTPS downloads, which honour the proxy variables.
- Do not print proxy values or other environment values in logs, reports or files.
- Never send document contents, project code or internal URLs to external services. Use
  external access only for dependencies and documentation.

## Experimenting with real models

The unit tests never load a BERT model. For experiments with real embeddings, use the locally
running service container (`docker ps`, container `radar-keyword-search`): it has torch,
transformers, spaCy and cached Hugging Face models (check
`/home/appuser/.cache/huggingface/hub`; this cache lives in the container, not in a volume,
so recreating the container means downloading the models again).

```bash
docker cp <scratch>/script.py radar-keyword-search:/tmp/script.py
docker exec -i radar-keyword-search python /tmp/script.py < input.txt > <scratch>/out.json
docker exec -u root radar-keyword-search rm -f /tmp/script.py   # docker cp files belong to root
```

- Scripts can import the service modules from `/app` (the image's version, which may lag
  behind the working tree).
- Do not install packages into the container. For extra packages, split the
  experiment: compute embeddings in the container, write JSON, and do the rest in the local
  Python 3.12 venv.
- Long German texts often exceed BERT's 512 tokens; the document embedding is then truncated.

For larger experiments, prefer the keyword benchmark (`evaluation/benchmark/`, see its
README): it runs both `candidates` options in-process in the local venv (after installing the
service's requirements) and scores them against reference and ChatGPT keywords.

## Licensing of dependencies

Check licenses before adding a dependency and record attribution duties in `NOTICE`.
Known case: the spaCy pipelines are MIT licensed, with training data credited in `NOTICE`.
