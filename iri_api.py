from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import asyncio
import httpx
import os

from urllib.parse import quote

app = FastAPI()

# -------------------------------
# Backend selection
# -------------------------------
# EXTRACTION_BACKEND picks which keyword-extraction model this server instance
# uses. Only that backend's dependencies are loaded, so choosing "ollama"
# never downloads the PubMedBERT model, and choosing "pubmedbert" never
# requires an Ollama server.
VALID_BACKENDS = {"pubmedbert", "ollama"}
EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "pubmedbert").strip().lower()
if EXTRACTION_BACKEND not in VALID_BACKENDS:
    raise RuntimeError(
        f"Invalid EXTRACTION_BACKEND '{EXTRACTION_BACKEND}'. "
        f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}."
    )

kw_model = None
llm_kw_model = None

# -------------------------------
# PubMedBERT Setup (CPU only)
# -------------------------------
if EXTRACTION_BACKEND == "pubmedbert":
    from transformers import AutoTokenizer, AutoModel
    from keybert import KeyBERT
    import torch

    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    class PubMedBERTEmbedding:
        def __call__(self, docs, **kwargs):
            encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            return embeddings.cpu().numpy()

    kw_model = KeyBERT(model=PubMedBERTEmbedding())

# -------------------------------
# LLM Setup: Ollama
# -------------------------------
# Ollama exposes an OpenAI-compatible chat API, so it's used via KeyBERT's
# OpenAI wrapper - just pointed at Ollama's endpoint with a placeholder key.
elif EXTRACTION_BACKEND == "ollama":
    from keybert import KeyLLM
    from keybert.llm import OpenAI as OpenAIWrapper
    import openai

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

    llm_wrapper = OpenAIWrapper(llm_client, model=llm_model, chat=True)
    llm_kw_model = KeyLLM(llm_wrapper)

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
    encoded_kw = quote(keyword)
    url = f"https://api.terminology.tib.eu/api/search?q={encoded_kw}"
    if ontology:
        url += f"&ontology={ontology}"

    if ontology_collection:
        url += f"&schema=collection&classification={ontology_collection}"


    print(f"TIB request url: {url}")

    try:
        response = await client.get(url, timeout=10)
        response.raise_for_status()
    except httpx.RequestError:
        return None

    data = response.json()
    best_match = None
    best_distance = float('inf')

    if "response" in data and "docs" in data["response"]:
        for doc in data["response"]["docs"]:
            iri = doc.get("iri")
            if not iri or not await check_iri_exists(iri, client):
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

    if best_match and best_match["distance"] <= threshold:
        return best_match
    return None

def _extract_keyword_list(document: str) -> list:
    """Extract a flat list of candidate keywords using the active backend."""
    if EXTRACTION_BACKEND == "pubmedbert":
        keyword_scores = kw_model.extract_keywords(
            document,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=10
        )
        return [kw for kw, _ in keyword_scores]

    # ollama: KeyLLM may return a flat list or a list-of-lists (one list per
    # input document), and can include blank entries.
    raw_keywords = llm_kw_model.extract_keywords(document)
    if raw_keywords and isinstance(raw_keywords[0], list):
        raw_keywords = raw_keywords[0]
    return [kw.strip() for kw in raw_keywords if kw and kw.strip()]

# -------------------------------
# Request Schema
# -------------------------------
class DocumentRequest(BaseModel):
    document: str
    ontology: Optional[str] = None
    ontology_collection: Optional[str] = None

# -------------------------------
# Endpoint
# -------------------------------
@app.post("/extract-iris")
async def extract_iris(req: DocumentRequest) -> Dict[str, Optional[Dict]]:
    try:
        print(f"Received request: {req}")

        keyword_list = _extract_keyword_list(req.document)
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

    except Exception as e:
        print("Unhandled error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "radar keyword service",
        "backend": EXTRACTION_BACKEND,
        "message": "Service is online",
    }


# -------------------------------
# Run the server
# -------------------------------
# Run with: uvicorn iri_api:app --reload
