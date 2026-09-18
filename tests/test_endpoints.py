from fastapi.testclient import TestClient

import iri_api


def _client(mod=iri_api):
    return TestClient(mod.app)


def test_health_check_reports_the_active_backend():
    with _client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "radar keyword service",
        "backend": "pubmedbert",
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


def test_extract_iris_via_ollama_backend_filters_blank_keywords(load_backend, monkeypatch):
    mod = load_backend("ollama")
    # The LLM can return blank/whitespace-only entries; those must be
    # filtered out and never sent to the TIB search.
    mod.llm_kw_model.extract_keywords.return_value = ["insulin", "  ", "glucose"]

    calls = []

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        calls.append((keyword, ontology, ontology_collection))
        return {"iri": f"https://example.org/{keyword}"}

    monkeypatch.setattr(mod, "search_tib_best_match", fake_search)

    with _client(mod) as client:
        response = client.post(
            "/extract-iris",
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


def test_extract_iris_via_ollama_backend(load_backend, monkeypatch):
    mod = load_backend("ollama")
    mod.llm_kw_model.extract_keywords.return_value = [["insulin"]]  # list-of-lists form

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return {"iri": f"https://example.org/{keyword}"}

    monkeypatch.setattr(mod, "search_tib_best_match", fake_search)

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin"})

    assert response.status_code == 200
    assert response.json() == {"insulin": {"iri": "https://example.org/insulin"}}


def test_extract_iris_via_ollama_backend_returns_500_when_extraction_fails(load_backend):
    mod = load_backend("ollama")

    def boom(*args, **kwargs):
        raise RuntimeError("llm exploded")

    mod.llm_kw_model.extract_keywords.side_effect = boom

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 500
    assert "llm exploded" in response.json()["detail"]


def test_extract_iris_via_ollama_backend_handles_no_keywords_found(load_backend):
    mod = load_backend("ollama")
    mod.llm_kw_model.extract_keywords.return_value = []

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "..."})

    assert response.status_code == 200
    assert response.json() == {}
