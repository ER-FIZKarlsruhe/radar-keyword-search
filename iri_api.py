from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import asyncio
import httpx
import os

from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
from keybert.llm import OpenAI as OpenAIWrapper
from keybert import KeyLLM
from urllib.parse import quote
import openai
import torch
import numpy as np

app = FastAPI()

# -------------------------------
# PubMedBERT Setup
# -------------------------------
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
# ChatGPT (OpenAI KeyBERT) Setup
# -------------------------------
OPENAI_API_KEY = os.getenv("CHAT_GPT_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("CHAT_GPT environment variable is not set.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
openai_llm = OpenAIWrapper(openai_client)
openai_kw_model = KeyLLM(openai_llm)

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

# -------------------------------
# Request Schema
# -------------------------------
class DocumentRequest(BaseModel):
    document: str
    ontology: Optional[str] = None
    ontology_collection: Optional[str] = None

# -------------------------------
# Endpoint: PubMedBERT
# -------------------------------
@app.post("/extract-iris")
async def extract_iris(req: DocumentRequest) -> Dict[str, Optional[Dict]]:
    try:
        keywords = kw_model.extract_keywords(
            req.document,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=10
        )

        ontology = req.ontology
        ontology_collection = req.ontology_collection
        print(f"Received request: {req}")

        async with httpx.AsyncClient() as client:
            tasks = [
                search_tib_best_match(kw, ontology, ontology_collection, HAMMING_THRESHOLD, client)
                for kw, _ in keywords
            ]
            results = await asyncio.gather(*tasks)

        best_matches = {kw: result for (kw, _), result in zip(keywords, results)}
        return best_matches

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Endpoint: ChatGPT (OpenAI)
# -------------------------------
@app.post("/extract-iris-openai")
async def extract_iris_openai(req: DocumentRequest) -> Dict[str, Optional[Dict]]:
    try:
        print("Received document:", req.document)

        try:
            keywords = openai_kw_model.extract_keywords(
                req.document
            )
            # Ensure it's a flat list of strings
            if isinstance(keywords[0], list):
                keywords = keywords[0]

            clean_keywords = [kw.strip() for kw in keywords if kw and kw.strip()]

        except Exception as llm_error:
            print("Keyword extraction failed:", llm_error)
            raise HTTPException(status_code=500, detail=f"Keyword extraction error: {str(llm_error)}")

        print("Extracted keywords:", keywords)

        ontology = req.ontology
        async with httpx.AsyncClient() as client:
            tasks = [
                search_tib_best_match(kw, ontology, HAMMING_THRESHOLD, client)
                for kw in clean_keywords
            ]
            results = await asyncio.gather(*tasks)

        best_matches = {kw: result for (kw), result in zip(keywords, results)}
        return best_matches

    except Exception as e:
        print("Unhandled error:", e)
        raise HTTPException(status_code=500, detail=str(e))
        

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "radar keyword service", "message": "Service is online"}


# -------------------------------
# Run the server
# -------------------------------
# Run with: uvicorn iri_api:app --reload
