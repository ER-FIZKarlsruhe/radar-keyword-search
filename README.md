
rd-search with TIB Terminology Service Support

This service provides keyword extraction from documents using either a custom PubMedBERT-based model or OpenAI's ChatGPT (via KeyBERT LLM), followed by entity linking to the [TIB Terminology Service](https://api.terminology.tib.eu).

---

## 🚀 Setup

### 1. Install Requirements

```bash
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

