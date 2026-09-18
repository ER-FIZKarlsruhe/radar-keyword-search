from fastapi.testclient import TestClient

import iri_api


def _client():
    return TestClient(iri_api.app)


def test_health_check():
    with _client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "radar keyword service",
        "message": "Service is online",
    }


def test_extract_iris_returns_a_match_per_extracted_keyword(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("insulin", 0.9)]

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return {"iri": "https://example.org/insulin"} if keyword == "insulin" else None

    monkeypatch.setattr(iri_api, "search_tib_best_match", fake_search)

    with _client() as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 200
    assert response.json() == {"insulin": {"iri": "https://example.org/insulin"}}


def test_extract_iris_returns_500_when_keyword_extraction_fails():
    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    iri_api.kw_model.extract_keywords.side_effect = boom

    with _client() as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 500
    assert "model exploded" in response.json()["detail"]


def test_extract_iris_openai_returns_a_match_per_cleaned_keyword(monkeypatch):
    # The LLM can return blank/whitespace-only entries; those must be
    # filtered out and never sent to the TIB search.
    iri_api.openai_kw_model.extract_keywords.return_value = ["insulin", "  ", "glucose"]

    calls = []

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        calls.append((keyword, ontology, ontology_collection))
        return {"iri": f"https://example.org/{keyword}"}

    monkeypatch.setattr(iri_api, "search_tib_best_match", fake_search)

    with _client() as client:
        response = client.post(
            "/extract-iris-openai",
            json={"document": "insulin and glucose", "ontology": "chebi", "ontology_collection": "cs"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "insulin": {"iri": "https://example.org/insulin"},
        "glucose": {"iri": "https://example.org/glucose"},
    }
    assert calls == [
        ("insulin", "chebi", "cs"),
        ("glucose", "chebi", "cs"),
    ]


def test_extract_iris_openai_handles_a_nested_keyword_list(monkeypatch):
    # KeyLLM sometimes returns a list-of-lists (one list per input document);
    # the endpoint flattens that to a single list of strings.
    iri_api.openai_kw_model.extract_keywords.return_value = [["insulin"]]

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return {"iri": f"https://example.org/{keyword}"}

    monkeypatch.setattr(iri_api, "search_tib_best_match", fake_search)

    with _client() as client:
        response = client.post("/extract-iris-openai", json={"document": "insulin"})

    assert response.status_code == 200
    assert response.json() == {"insulin": {"iri": "https://example.org/insulin"}}


def test_extract_iris_openai_returns_500_when_keyword_extraction_fails():
    def boom(*args, **kwargs):
        raise RuntimeError("llm exploded")

    iri_api.openai_kw_model.extract_keywords.side_effect = boom

    with _client() as client:
        response = client.post("/extract-iris-openai", json={"document": "insulin study"})

    assert response.status_code == 500
    assert "llm exploded" in response.json()["detail"]
