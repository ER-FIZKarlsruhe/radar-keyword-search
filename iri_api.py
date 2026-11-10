# Copyright 2025-2026 FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time


def _startup_log(message: str) -> None:
    """Print a timestamped startup message to stderr, flushed at once.

    Startup (imports + loading the default model) runs before uvicorn's own logging
    starts, and stdout is block-buffered in a container, so plain print()s only showed
    up much later - or never, while the startup hangs, which is when they are needed.
    """
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [startup] {message}", file=sys.stderr, flush=True)


_startup_log(f"Starting radar keyword service (pid {os.getpid()}), importing libraries...")

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional
import uvicorn
import asyncio
import httpx
import threading

from urllib.parse import quote

import torch
from transformers import CONFIG_MAPPING, AutoModel, AutoTokenizer

_startup_log("Libraries imported.")

from bert_models import BERT_MODEL_REGISTRY
from keyword_extraction import KeywordExtractor
from stop_words import STOP_WORDS

app = FastAPI()

# -------------------------------
# Keyword extraction backend
# -------------------------------
# Keywords are extracted with BERT models only (the KeyBERT algorithm, see
# keyword_extraction.py). The former LLM backend (EXTRACTION_BACKEND=keyllm,
# via Ollama) was removed. A deployment that still asks for it fails at startup
# instead of silently getting BERT keywords.
EXTRACTION_BACKEND = "keybert"
_requested_backend = os.getenv("EXTRACTION_BACKEND", EXTRACTION_BACKEND).strip().lower()
if _requested_backend != EXTRACTION_BACKEND:
    raise RuntimeError(
        f"Invalid EXTRACTION_BACKEND '{_requested_backend}'. Only '{EXTRACTION_BACKEND}' is "
        f"supported; the LLM backend 'keyllm' was removed. Unset EXTRACTION_BACKEND."
    )

# -------------------------------
# BERT model registry
# -------------------------------
# Defined in bert_models.py; which specific model a request uses is selected
# per-request via DocumentRequest.model.
DEFAULT_BERT_MODEL = os.getenv("BERT_MODEL", "pubmedbert").strip().lower()
if DEFAULT_BERT_MODEL not in BERT_MODEL_REGISTRY:
    raise RuntimeError(
        f"Invalid BERT_MODEL '{DEFAULT_BERT_MODEL}'. "
        f"Must be one of: {', '.join(sorted(BERT_MODEL_REGISTRY))}."
    )

# Loaded lazily (see get_bert_model) and cached here, keyed by registry name,
# so a request for a non-default model only pays the load cost once.
_bert_model_cache: Dict[str, object] = {}

# -------------------------------
# BERT Setup (CPU only)
# -------------------------------
# Position-embedding limit shared by every BERT-style model in BERT_MODEL_REGISTRY.
BERT_MAX_TOKENS = 512

# Every distinct word of a document is a candidate keyword and gets embedded. Embedding
# them in batches of this size keeps memory bounded however long the document is.
EMBED_BATCH_SIZE = 64

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class _BertEmbedding:
    """Mean-pooled embeddings from a BERT_MODEL_REGISTRY model; KeywordExtractor's embed function."""

    def __init__(self, model_name: str, prefix: str = "", model_type: Optional[str] = None):
        # model_type is only set for older checkpoints whose config.json doesn't name their
        # architecture (see bert_models.py): their config is then loaded with that type's
        # config class, and both Auto classes pick the tokenizer/model from it.
        kwargs = {}
        if model_type:
            kwargs["config"] = CONFIG_MAPPING[model_type].from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self._model = AutoModel.from_pretrained(model_name, **kwargs)
        self._prefix = prefix

    def embed(self, docs):
        # KeywordExtractor passes the candidate words as a numpy array of str, which the
        # tokenizer rejects - it only accepts a str or a list of str.
        docs = [self._prefix + str(doc) for doc in docs]
        batches = [self._embed_batch(docs[i:i + EMBED_BATCH_SIZE]) for i in range(0, len(docs), EMBED_BATCH_SIZE)]
        return torch.cat(batches).cpu().numpy()

    def _embed_batch(self, docs):
        # Explicit max_length: some checkpoints (e.g. SciBERT) don't set model_max_length,
        # so truncation=True alone wouldn't truncate and a long abstract would overflow
        # BERT's 512 position embeddings.
        encoded_input = self._tokenizer(
            docs, padding=True, truncation=True, max_length=BERT_MAX_TOKENS, return_tensors='pt'
        )
        with torch.no_grad():
            model_output = self._model(**encoded_input)
        return _mean_pooling(model_output, encoded_input['attention_mask'])

def get_bert_model(model_key: str) -> "KeywordExtractor":
    """Return the KeywordExtractor for model_key, loading and caching it on first use."""
    if model_key not in BERT_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown BERT model '{model_key}'. "
            f"Must be one of: {', '.join(sorted(BERT_MODEL_REGISTRY))}."
        )
    if model_key not in _bert_model_cache:
        entry = BERT_MODEL_REGISTRY[model_key]
        hf_model = entry["hf_model"]
        _startup_log(f"Loading BERT model '{model_key}' ({hf_model})...")
        started = time.monotonic()
        _bert_model_cache[model_key] = KeywordExtractor(
            _BertEmbedding(hf_model, entry.get("prefix", ""), entry.get("model_type")).embed
        )
        _startup_log(f"BERT model '{model_key}' loaded in {time.monotonic() - started:.1f}s.")
    return _bert_model_cache[model_key]

# Eagerly load the default model at startup so the first request isn't
# slowed down by an on-demand load. Non-default models are loaded lazily,
# the first time a request asks for them. If the model isn't in the Hugging Face
# cache yet it is downloaded here, which hangs without internet access - so log
# where it comes from and through which proxy.
_startup_log(
    f"Default BERT model '{DEFAULT_BERT_MODEL}'; Hugging Face cache "
    f"{os.getenv('HF_HOME', '~/.cache/huggingface')}, HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE', 'unset')}, "
    f"HTTPS_PROXY {'set' if os.getenv('HTTPS_PROXY') or os.getenv('https_proxy') else 'unset'}"
)
kw_model = get_bert_model(DEFAULT_BERT_MODEL)
_startup_log("Startup complete, uvicorn will now open the port.")

# -------------------------------
# Supporting Functions
# -------------------------------
HAMMING_THRESHOLD = 5

# A match may differ from the keyword in at most one character per this many keyword
# characters. Without it, the fixed threshold of 5 accepted *any* term for a keyword of
# up to 5 letters ("hier" -> "usko") and long prefix matches ("witches" -> "Witches game");
# with it, short keywords need an exact match and only small edits ("nanoparticles" ->
# "nanoparticle") pass.
CHARS_PER_ALLOWED_DIFFERENCE = 4


def allowed_distance(keyword: str, threshold: int) -> int:
    """Largest Hamming distance at which `keyword` counts as matched: a quarter of its
    length (rounded down), and never more than `threshold` (the request's max_distance)."""
    return min(threshold, len(keyword.strip()) // CHARS_PER_ALLOWED_DIFFERENCE)


def hamming_distance(s1, s2):
    max_len = max(len(s1), len(s2))
    s1 = s1.ljust(max_len)
    s2 = s2.ljust(max_len)
    return sum(el1 != el2 for el1, el2 in zip(s1.lower(), s2.lower()))

# Only these answers prove an IRI dead. Anything else is no proof either way and keeps the match:
# hosts behind bot protection answer automated requests with 403 (www.loc.gov, which hosts the
# PREMIS / premis3 terms), and after a burst of requests also with a challenge (503) or by
# stalling, so requiring a 200 dropped every PREMIS term, depending on the protection's mood.
IRI_DEAD_STATUS_CODES = {404, 410}


async def check_iri_exists(iri, client: httpx.AsyncClient) -> bool:
    """False only if the IRI is definitely dead: answered with 404/410, or its host can't be
    resolved / refuses the connection. Other statuses and timeouts don't disprove it."""
    try:
        response = await client.head(iri, follow_redirects=True, timeout=5)
    except httpx.TimeoutException:
        print(f"IRI check timed out, keeping the match: {iri}", flush=True)
        return True
    except httpx.RequestError as e:
        print(f"IRI check failed ({type(e).__name__}), dropping the match: {iri}", flush=True)
        return False
    if response.status_code != 200:
        print(f"IRI check status {response.status_code}: {iri}", flush=True)
    return response.status_code not in IRI_DEAD_STATUS_CODES

TIB_COLLECTIONS_URL = "https://api.terminology.tib.eu/api/ontologies/schemavalues?schema=collection"

# lowercase id -> TIB's exact id; None until fetched successfully once
_tib_collection_ids: Optional[Dict[str, str]] = None


async def _get_tib_collection_ids(client: httpx.AsyncClient) -> Dict[str, str]:
    global _tib_collection_ids
    if _tib_collection_ids is None:
        try:
            response = await client.get(TIB_COLLECTIONS_URL, timeout=10)
            response.raise_for_status()
            ids = response.json().get("content") or []
            mapping = {i.lower(): i for i in ids if isinstance(i, str)}
            # Only cache a real list, so a transient failure is retried on the next request
            if mapping:
                _tib_collection_ids = mapping
            return mapping
        except (httpx.HTTPError, ValueError):
            return {}
    return _tib_collection_ids


async def resolve_collection_ids(ontology_collection: Optional[str], client: httpx.AsyncClient) -> Optional[str]:
    """Map a (comma-separated) collection value to TIB's exact ids.

    TIB's classification filter is case-sensitive and matches only the exact id,
    which is often mixed case ("DataPLANT", "NFDI4Energy", "FAIR Data Spaces").
    Any other casing silently matches nothing - and, with no `ontology` set,
    the whole term lookup then returns zero candidates. So each id is looked up
    case-insensitively in TIB's own collection list; an id TIB doesn't list (or
    if the list can't be fetched) is uppercased, which is right for the common
    all-uppercase ids like "NFDI4CHEM".
    """
    if not ontology_collection:
        return ontology_collection
    parts = [p.strip() for p in ontology_collection.split(",") if p.strip()]
    if not parts:
        return None
    known = await _get_tib_collection_ids(client)
    return ",".join(known.get(p.lower(), p.upper()) for p in parts)


async def search_tib_best_match(keyword: str, ontology: Optional[str],  ontology_collection: Optional[str], threshold: int, client: httpx.AsyncClient) -> Optional[Dict]:
    """Find the TIB term closest to `keyword` and report how close it was.

    Always returns the globally closest candidate (with its distance and a
    `matched` flag for whether it cleared `threshold`) as long as at least one
    candidate was found at all, so callers can show a match-quality rating
    even for keywords that end up used as free text. Returns None only when
    the TIB search itself failed or returned no candidates whatsoever.
    """
    encoded_kw = quote(keyword)
    url = f"https://api.terminology.tib.eu/api/search?q={encoded_kw}"
    if ontology:
        url += f"&ontology={ontology}"

    if ontology_collection:
        # Expected to already be in TIB's exact casing - see resolve_collection_ids.
        url += f"&schema=collection&classification={quote(ontology_collection, safe=',')}"


    print(f"TIB request url: {url}")

    try:
        response = await client.get(url, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        # Covers both connection-level failures (httpx.RequestError) and
        # non-2xx responses (httpx.HTTPStatusError from raise_for_status()).
        # A single term's TIB lookup failing should not fail the whole batch
        # in extract_iris's asyncio.gather - skip just this term instead.
        return None

    data = response.json()
    best_match = None
    best_distance = float('inf')

    # Textual closeness is ranked across every candidate term first, regardless of whether its
    # IRI currently resolves - checking that per-candidate up front (as this used to do) meant a
    # perfect (distance 0) label match got silently discarded whenever its IRI happened to be
    # dead, even though the match quality itself had nothing to do with that.
    if "response" in data and "docs" in data["response"]:
        for doc in data["response"]["docs"]:
            iri = doc.get("iri")
            if not iri:
                continue

            terms = []
            if "label" in doc:
                terms.append(doc["label"])
            if "synonym" in doc:
                terms.extend(doc["synonym"])

            for term in terms:
                dist = hamming_distance(keyword, term)
                if dist < best_distance:
                    best_distance = dist
                    scheme_uri, last_segment = _split_iri(iri)
                    best_match = {
                        "iri": iri,
                        "label": doc.get("label"),
                        "best_term": term,
                        "distance": dist,
                        "ontology_name": doc.get("ontology_name"),
                        "source": "tib",
                        "keyword_scheme": KEYWORD_SCHEME_OTHER,
                        "scheme_uri": scheme_uri,
                        "classification_code": doc.get("short_form") or last_segment,
                    }

    if not best_match:
        return None

    # Only the winning candidate's IRI needs a liveness check (also cheaper than checking every
    # candidate up front, as before). A match can still be "matched" quality-wise while lacking a
    # usable link, if that one IRI doesn't resolve.
    best_match["matched"] = (
        best_match["distance"] <= allowed_distance(keyword, threshold)
        and await check_iri_exists(best_match["iri"], client)
    )
    return best_match


# -------------------------------
# GND (Gemeinsame Normdatei) via lobid-gnd
# -------------------------------
# lobid-gnd (hbz) is a JSON search API over the GND authority file; it's also
# what radar-frontend's interactive GND keyword typeahead uses (gndManager.js).
GND_SEARCH_URL = os.getenv("GND_SEARCH_URL", "https://lobid.org/gnd/search")
GND_SCHEME_URI = "https://d-nb.info/gnd/"
GND_SEARCH_SIZE = 20

# GND entity types searched, in order of preference. Without a type filter, a plain-text GND
# search is dominated by works and conferences (e.g. "Goethe" returns five *Works* titled
# "Goethe" first), which is noise for keywords. The order matters: when candidates of
# different types are equally close, the type listed first wins, so e.g. the subject heading
# "Faust" beats a person whose name is just "Faust".
GND_TYPES = [
    t.strip()
    for t in os.getenv("GND_TYPES", "SubjectHeading,PlaceOrGeographicName,CorporateBody,Person").split(",")
    if t.strip()
]

# A single word can't identify a person: "napoleon" or "goethe" would otherwise match some
# person whose (variant) name is just "Napoleon" or "Goethe" exactly - see _gnd_person_terms.
GND_PERSON_TYPE = "Person"

# Values of RADAR's keywordScheme enum (radar-common RadarTypes.xsd)
KEYWORD_SCHEME_GND = "GND"
KEYWORD_SCHEME_OTHER = "Other"


def _split_iri(iri: str):
    """Split an IRI into (scheme URI incl. trailing '/', last path segment) - the same split
    radar-frontend applies to TIB IRIs for a keyword's schemeURI/classificationCode."""
    idx = iri.rfind("/")
    if idx == -1:
        return None, iri
    return iri[: idx + 1], iri[idx + 1:]


def _gnd_type_rank(member: Dict) -> int:
    """Position of the member's most preferred type in GND_TYPES (lower is better)."""
    types = set(member.get("type") or [])
    return next((i for i, t in enumerate(GND_TYPES) if t in types), len(GND_TYPES))


def _natural_name_order(name: str) -> Optional[str]:
    """'Goethe, Johann Wolfgang von' -> 'Johann Wolfgang von Goethe'; None if not inverted."""
    parts = [p.strip() for p in name.split(",")]
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[1]} {parts[0]}"


def _gnd_candidate_terms(keyword: str, member: Dict) -> list:
    """The names of a lobid member to compare `keyword` against.

    Persons are only considered for multi-word keywords, and their inverted names
    ("Surname, Forenames") are also compared in natural order, so an extracted
    "Johann Wolfgang von Goethe" can match the GND's "Goethe, Johann Wolfgang von".
    """
    names = [member["preferredName"], *(member.get("variantName") or [])]
    if GND_PERSON_TYPE not in (member.get("type") or []):
        return names
    if len(keyword.split()) < 2:
        return []
    terms = []
    for name in names:
        terms.append(name)
        natural = _natural_name_order(name)
        if natural:
            terms.append(natural)
    return terms


async def search_gnd_best_match(keyword: str, threshold: int, client: httpx.AsyncClient) -> Optional[Dict]:
    """Find the GND entity closest to `keyword`, analogous to search_tib_best_match.

    Same return contract: the closest candidate (by Hamming distance over the preferred
    name and all variant names; on a tie, the type listed first in GND_TYPES wins) with a
    `matched` flag, or None if the search failed or found nothing. Unlike TIB, no extra HEAD request is made against the winning IRI:
    lobid only returns live GND records, and https://d-nb.info/gnd/... merely redirects to
    the DNB portal's HTML search page, so a liveness check would add nothing but load.
    """
    params = {"q": keyword, "format": "json", "size": GND_SEARCH_SIZE}
    # Persons can't match a single word (see _gnd_candidate_terms), so don't let them fill the
    # result page and push out the subject headings that could
    types = [t for t in GND_TYPES if t != GND_PERSON_TYPE or len(keyword.split()) >= 2]
    if types:
        params["filter"] = " OR ".join(f"type:{t}" for t in types)
    elif GND_TYPES:
        return None

    try:
        response = await client.get(GND_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        # See search_tib_best_match: one failing term must not fail the whole batch.
        return None

    best_match = None
    best_key = (float("inf"), float("inf"))
    for member in data.get("member") or []:
        iri = member.get("id")
        preferred = member.get("preferredName")
        if not iri or not preferred:
            continue
        rank = _gnd_type_rank(member)
        for term in _gnd_candidate_terms(keyword, member):
            dist = hamming_distance(keyword, term)
            if (dist, rank) < best_key:
                best_key = (dist, rank)
                best_match = {
                    "iri": iri,
                    "label": preferred,
                    "best_term": term,
                    "distance": dist,
                    "ontology_name": None,
                    "source": "gnd",
                    "keyword_scheme": KEYWORD_SCHEME_GND,
                    "scheme_uri": GND_SCHEME_URI,
                    "classification_code": member.get("gndIdentifier") or _split_iri(iri)[1],
                }

    if not best_match:
        return None
    best_match["matched"] = best_match["distance"] <= allowed_distance(keyword, threshold)
    return best_match


def pick_better_match(a: Optional[Dict], b: Optional[Dict]) -> Optional[Dict]:
    """Choose between two candidates for one keyword (terminology="both").

    A matched candidate beats an unmatched one, then the smaller distance wins. On a tie
    the first argument wins - extract_iris passes TIB first, since its ontologies are the
    ones scoped to the workspace's domain.
    """
    if a is None or b is None:
        return a or b
    if a["matched"] != b["matched"]:
        return a if a["matched"] else b
    return a if a["distance"] <= b["distance"] else b

async def _none() -> None:
    return None


DEFAULT_MAX_KEYWORDS = 10

# Longest accepted document. Real requests are far shorter: radar-frontend sends title +
# descriptions + optionally a README cut to 20,000 characters, and the longest of 708
# harvested RADAR documents has 7,464. The cap bounds the work one request can cause.
MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "100000"))

# One extraction at a time: a Hugging Face fast tokenizer must not be used from two
# threads at once ("Already borrowed"), torch already uses every CPU core for one
# extraction, and a model requested twice at once must be loaded only once. Requests
# queue here, in a worker thread, while the event loop keeps serving everything else.
_extraction_lock = threading.Lock()

def _extract_keyword_list(document: str, bert_model: Optional[str] = None, max_keywords: int = DEFAULT_MAX_KEYWORDS) -> list:
    """Extract a flat list of keywords, most relevant first.

    bert_model selects a specific BERT_MODEL_REGISTRY entry (default:
    DEFAULT_BERT_MODEL). max_keywords caps how many keywords are returned;
    there may be fewer if the document doesn't yield that many candidates.
    """
    with _extraction_lock:
        model = get_bert_model(bert_model or DEFAULT_BERT_MODEL)
        keyword_scores = model.extract_keywords(
            document,
            top_n=max_keywords,
            keyphrase_ngram_range=(1, 1),
            stop_words=STOP_WORDS,
        )
    return [kw for kw, _ in keyword_scores]

# -------------------------------
# Request Schema
# -------------------------------
class DocumentRequest(BaseModel):
    document: str = Field(max_length=MAX_DOCUMENT_CHARS)
    ontology: Optional[str] = None
    ontology_collection: Optional[str] = None
    model: Optional[str] = None
    max_keywords: Optional[int] = Field(default=None, ge=1, le=20)
    # Where the extracted keywords are looked up: the TIB Terminology Service, the GND,
    # or both (the better match per keyword wins). ontology/ontology_collection only
    # apply to TIB.
    terminology: Literal["tib", "gnd", "both"] = "tib"
    # Accept a match only up to this Hamming distance; 0 = exact matches only (ignoring
    # case). Defaults to HAMMING_THRESHOLD. Shorter keywords get less (see
    # allowed_distance). Closer candidates are still reported with matched=false, so
    # clients can show how close they were.
    max_distance: Optional[int] = Field(default=None, ge=0, le=HAMMING_THRESHOLD)

# -------------------------------
# Endpoint
# -------------------------------
@app.post("/extract-iris")
async def extract_iris(req: DocumentRequest) -> Dict[str, Optional[Dict]]:
    try:
        print(f"Received request: {req}")

        if req.model and req.model not in BERT_MODEL_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown BERT model '{req.model}'. "
                    f"Must be one of: {', '.join(sorted(BERT_MODEL_REGISTRY))}."
                ),
            )

        # CPU-bound: run it in a worker thread so the event loop stays free for other
        # requests (health checks, the TIB/GND lookups of requests already past this point).
        keyword_list = await run_in_threadpool(
            _extract_keyword_list, req.document, req.model, req.max_keywords or DEFAULT_MAX_KEYWORDS
        )
        print(f"Extracted keywords: {keyword_list}")

        threshold = HAMMING_THRESHOLD if req.max_distance is None else req.max_distance
        use_tib = req.terminology in ("tib", "both")
        use_gnd = req.terminology in ("gnd", "both")

        async def lookup(kw: str, ontology_collection: Optional[str]) -> Optional[Dict]:
            tib, gnd = await asyncio.gather(
                search_tib_best_match(kw, req.ontology, ontology_collection, threshold, client) if use_tib else _none(),
                search_gnd_best_match(kw, threshold, client) if use_gnd else _none(),
            )
            return pick_better_match(tib, gnd)

        async with httpx.AsyncClient() as client:
            ontology_collection = await resolve_collection_ids(req.ontology_collection, client) if use_tib else None
            results = await asyncio.gather(*(lookup(kw, ontology_collection) for kw in keyword_list))

        return dict(zip(keyword_list, results))

    except HTTPException:
        raise
    except Exception as e:
        print("Unhandled error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List the BERT models available for /extract-iris's `model` parameter."""
    return {
        "backend": EXTRACTION_BACKEND,
        "default_model": DEFAULT_BERT_MODEL,
        "models": [
            {
                "name": name,
                "hf_model": entry["hf_model"],
                "label": entry["label"],
                "kind": entry["kind"],
                "languages": entry["languages"],
                "domains": entry["domains"],
                "default": name == DEFAULT_BERT_MODEL,
                "loaded": name in _bert_model_cache,
            }
            for name, entry in sorted(BERT_MODEL_REGISTRY.items())
        ],
    }


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "radar keyword service",
        "backend": EXTRACTION_BACKEND,
        "default_bert_model": DEFAULT_BERT_MODEL,
        "message": "Service is online",
    }


# -------------------------------
# Run the server
# -------------------------------
# Run with: uvicorn iri_api:app --reload
