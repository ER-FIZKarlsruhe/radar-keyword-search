from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import uvicorn
import asyncio
import httpx
import json
import os

from urllib.parse import quote

app = FastAPI()

# -------------------------------
# Backend selection
# -------------------------------
# EXTRACTION_BACKEND picks which keyword-extraction model this server instance
# uses. Only that backend's dependencies are loaded, so choosing "keyllm"
# never downloads a BERT model, and choosing "keybert" never requires an
# Ollama server. Which specific BERT model "keybert" serves is a separate,
# per-request choice - see BERT_MODEL_REGISTRY below.
VALID_BACKENDS = {"keybert", "keyllm"}
EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "keybert").strip().lower()
if EXTRACTION_BACKEND not in VALID_BACKENDS:
    raise RuntimeError(
        f"Invalid EXTRACTION_BACKEND '{EXTRACTION_BACKEND}'. "
        f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}."
    )

kw_model = None
llm_client = None
llm_model = None

# -------------------------------
# BERT model registry
# -------------------------------
# Alternative embedding models the "keybert" backend can serve, selectable
# per-request via DocumentRequest.model. They're all BERT-family encoders
# loaded through the same AutoTokenizer/AutoModel + mean-pooling path, so
# adding a new one only means adding an entry here.
BERT_MODEL_REGISTRY = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "pubmedbert-large": "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "scibert": "allenai/scibert_scivocab_uncased",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}
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
if EXTRACTION_BACKEND == "keybert":
    from transformers import AutoTokenizer, AutoModel
    from keybert import KeyBERT
    import torch

    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    class _BertEmbedding:
        def __init__(self, model_name: str):
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)

        def __call__(self, docs, **kwargs):
            encoded_input = self._tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self._model(**encoded_input)
            embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
            return embeddings.cpu().numpy()

    def get_bert_model(model_key: str) -> "KeyBERT":
        """Return the KeyBERT instance for model_key, loading and caching it on first use."""
        if model_key not in BERT_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown BERT model '{model_key}'. "
                f"Must be one of: {', '.join(sorted(BERT_MODEL_REGISTRY))}."
            )
        if model_key not in _bert_model_cache:
            print(f"Loading BERT model '{model_key}' ({BERT_MODEL_REGISTRY[model_key]})...")
            _bert_model_cache[model_key] = KeyBERT(model=_BertEmbedding(BERT_MODEL_REGISTRY[model_key]))
        return _bert_model_cache[model_key]

    # Eagerly load the default model at startup so the first request isn't
    # slowed down by an on-demand load. Non-default models are loaded lazily,
    # the first time a request asks for them.
    kw_model = get_bert_model(DEFAULT_BERT_MODEL)

# -------------------------------
# LLM Setup: Ollama
# -------------------------------
# Ollama exposes an OpenAI-compatible chat API. This talks to it directly
# with the `openai` client rather than through KeyBERT's KeyLLM/OpenAI
# wrapper: KeyLLM's extract_keywords() always does a naive
# response.choices[0].message.content.split(",") on the raw reply, so a chat
# model that ignores the "respond with ONLY the keywords" instruction and
# answers conversationally (e.g. "Sure! Here are the keywords: ...") leaks
# its lead-in/sign-off sentences through as bogus keywords - no prompt
# wording closed that gap completely. Requesting response_format=
# {"type": "json_object"} instead constrains the model's output to valid
# JSON via grammar-constrained decoding, so there's no free-form prose left
# for a leak to hide in, and parsing it doesn't depend on the model
# following an informal "separated by commas" instruction at all.
elif EXTRACTION_BACKEND == "keyllm":
    import openai

    OLLAMA_KEYWORD_EXTRACTION_SYSTEM_PROMPT = (
        "You extract keywords from text. Respond with ONLY a JSON object of the exact "
        'shape {"keywords": ["keyword1", "keyword2", ...]} and nothing else - no '
        "commentary, explanations, or introductory phrases."
    )

    OLLAMA_KEYWORD_EXTRACTION_PROMPT = """I have the following document:
[DOCUMENT]

Extract at most [MAX_KEYWORDS] keywords that best describe the topic of the text.
Respond with a JSON object of the exact shape {"keywords": ["keyword1", "keyword2", ...]}."""

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    llm_model = os.getenv("OLLAMA_MODEL", "llama3")
    # Ollama doesn't check the API key, but the OpenAI client requires one.
    # Ollama is typically local/internal, and by default the openai client's
    # httpx transport honours HTTP_PROXY/HTTPS_PROXY/NO_PROXY from the
    # environment - on a machine behind a corporate proxy, that can route
    # these requests through the proxy and break them. So by default we
    # bypass any such proxy entirely; set OLLAMA_HTTP_PROXY to route through
    # a specific proxy instead (e.g. if Ollama itself is only reachable
    # through one).
    ollama_proxy = os.getenv("OLLAMA_HTTP_PROXY")
    ollama_http_client = httpx.Client(proxy=ollama_proxy) if ollama_proxy else httpx.Client(trust_env=False)
    llm_client = openai.OpenAI(api_key="ollama", base_url=base_url, http_client=ollama_http_client)

# -------------------------------
# Supporting Functions
# -------------------------------
HAMMING_THRESHOLD = 5

def hamming_distance(s1, s2):
    max_len = max(len(s1), len(s2))
    s1 = s1.ljust(max_len)
    s2 = s2.ljust(max_len)
    return sum(el1 != el2 for el1, el2 in zip(s1.lower(), s2.lower()))

async def check_iri_exists(iri, client: httpx.AsyncClient) -> bool:
    try:
        response = await client.head(iri, follow_redirects=True, timeout=5)
        return response.status_code == 200
    except httpx.RequestError:
        return False

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
        # TIB's classification filter is case-sensitive and only recognizes the
        # uppercase collection id (e.g. "NFDI4CHEM"); a lowercase/mixed-case value
        # is silently treated as unrecognized and matches nothing, with no error.
        url += f"&schema=collection&classification={ontology_collection.upper()}"


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
                    best_match = {
                        "iri": iri,
                        "label": doc.get("label"),
                        "best_term": term,
                        "distance": dist,
                        "ontology_name": doc.get("ontology_name")
                    }

    if not best_match:
        return None

    # Only the winning candidate's IRI needs a liveness check (also cheaper than checking every
    # candidate up front, as before). A match can still be "matched" quality-wise while lacking a
    # usable link, if that one IRI doesn't resolve.
    best_match["matched"] = best_match["distance"] <= threshold and await check_iri_exists(best_match["iri"], client)
    return best_match

DEFAULT_MAX_KEYWORDS = 10

def _extract_keywords_via_ollama(document: str, max_keywords: int = DEFAULT_MAX_KEYWORDS) -> list:
    """Extract keywords from the configured Ollama chat model as a JSON array.

    Calls the OpenAI-compatible client directly with
    response_format={"type": "json_object"} instead of going through
    KeyBERT's KeyLLM (see the module-level comment above the Ollama setup for
    why): the model can't leak conversational prose through the result
    because it's constrained to emit valid JSON in the first place.

    max_keywords is only a best-effort instruction to the chat model (unlike
    KeyBERT's top_n, nothing here can force an exact count), so the result is
    still truncated afterwards to honour the caller's limit.
    """
    prompt = OLLAMA_KEYWORD_EXTRACTION_PROMPT.replace("[DOCUMENT]", document).replace(
        "[MAX_KEYWORDS]", str(max_keywords)
    )
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": OLLAMA_KEYWORD_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    keywords = parsed.get("keywords", []) if isinstance(parsed, dict) else []
    return [str(kw).strip() for kw in keywords if str(kw).strip()][:max_keywords]

def _extract_keyword_list(document: str, bert_model: Optional[str] = None, max_keywords: int = DEFAULT_MAX_KEYWORDS) -> list:
    """Extract a flat list of candidate keywords using the active backend.

    bert_model selects a specific BERT_MODEL_REGISTRY entry for the "keybert"
    backend; it's ignored (and meaningless) for "keyllm". max_keywords caps how
    many keywords are returned; there may be fewer if the document doesn't
    yield that many candidates.
    """
    if EXTRACTION_BACKEND == "keybert":
        model = get_bert_model(bert_model or DEFAULT_BERT_MODEL)
        keyword_scores = model.extract_keywords(
            document,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=max_keywords
        )
        return [kw for kw, _ in keyword_scores]

    return _extract_keywords_via_ollama(document, max_keywords)

# -------------------------------
# Request Schema
# -------------------------------
class DocumentRequest(BaseModel):
    document: str
    ontology: Optional[str] = None
    ontology_collection: Optional[str] = None
    model: Optional[str] = None
    max_keywords: Optional[int] = Field(default=None, ge=1, le=20)

# -------------------------------
# Endpoint
# -------------------------------
@app.post("/extract-iris")
async def extract_iris(req: DocumentRequest) -> Dict[str, Optional[Dict]]:
    try:
        print(f"Received request: {req}")

        if req.model:
            if EXTRACTION_BACKEND != "keybert":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'model' is only supported when EXTRACTION_BACKEND=keybert "
                        f"(current backend: '{EXTRACTION_BACKEND}')."
                    ),
                )
            if req.model not in BERT_MODEL_REGISTRY:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown BERT model '{req.model}'. "
                        f"Must be one of: {', '.join(sorted(BERT_MODEL_REGISTRY))}."
                    ),
                )

        keyword_list = _extract_keyword_list(req.document, req.model, req.max_keywords or DEFAULT_MAX_KEYWORDS)
        print(f"Extracted keywords: {keyword_list}")

        ontology = req.ontology
        ontology_collection = req.ontology_collection

        async with httpx.AsyncClient() as client:
            tasks = [
                search_tib_best_match(kw, ontology, ontology_collection, HAMMING_THRESHOLD, client)
                for kw in keyword_list
            ]
            results = await asyncio.gather(*tasks)

        return dict(zip(keyword_list, results))

    except HTTPException:
        raise
    except Exception as e:
        print("Unhandled error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List the BERT models available for /extract-iris's `model` parameter.

    Only meaningful when EXTRACTION_BACKEND=keybert; returned regardless so
    clients can discover what switching backends would offer.
    """
    return {
        "backend": EXTRACTION_BACKEND,
        "default_model": DEFAULT_BERT_MODEL,
        "models": [
            {
                "name": name,
                "hf_model": hf_id,
                "default": name == DEFAULT_BERT_MODEL,
                "loaded": name in _bert_model_cache,
            }
            for name, hf_id in sorted(BERT_MODEL_REGISTRY.items())
        ],
    }


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "radar keyword service",
        "backend": EXTRACTION_BACKEND,
        "default_bert_model": DEFAULT_BERT_MODEL if EXTRACTION_BACKEND == "keybert" else None,
        "message": "Service is online",
    }


# -------------------------------
# Run the server
# -------------------------------
# Run with: uvicorn iri_api:app --reload
