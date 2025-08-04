from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn

from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
import torch
import numpy as np
import requests

# FastAPI app
app = FastAPI()

# Load PubMedBERT
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Custom embedding function
class PubMedBERTEmbedding:
    def __call__(self, docs, **kwargs):
        encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy()

# KeyBERT with custom embedder
kw_model = KeyBERT(model=PubMedBERTEmbedding())

# Threshold for Hamming distance
HAMMING_THRESHOLD = 5

# Hamming distance helper
def hamming_distance(s1, s2):
    max_len = max(len(s1), len(s2))
    s1 = s1.ljust(max_len)
    s2 = s2.ljust(max_len)
    return sum(el1 != el2 for el1, el2 in zip(s1.lower(), s2.lower()))

# Check if IRI is valid
def check_iri_exists(iri):
    try:
        response = requests.head(iri, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# TIB IRI lookup
def search_tib_best_match(keyword, ontology, threshold):
    url = f"https://api.terminology.tib.eu/api/search?q={keyword}"

    if ontology:
            url += f"&ontology={ontology}"

    print(f"TIB request url: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    best_match = None
    best_distance = float('inf')

    if "response" in data and "docs" in data["response"]:
        for doc in data["response"]["docs"]:
            iri = doc.get("iri")
            if not iri or not check_iri_exists(iri):
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

# Request model
class DocumentRequest(BaseModel):
    document: str
    ontology: Optional[str] = None

# API endpoint
@app.post("/extract-iris")
def extract_iris(req: DocumentRequest) -> Dict[str, Optional[Dict]]:
    try:
        keywords = kw_model.extract_keywords(
            req.document,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=10
        )

        best_matches = {}
        ontology = getattr(req, 'ontology', None)
        print(f"Received request: {req}")

        print(f"Document: {req.document}")
        print(f"Ontology: {getattr(req, 'ontology', None)}")


        for kw, _ in keywords:
            match = search_tib_best_match(kw, ontology, HAMMING_THRESHOLD)
            best_matches[kw] = match

        return best_matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn filename:app --reload

