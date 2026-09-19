from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import iri_api


def _client(mod=iri_api):
    return TestClient(mod.app)


def _fake_chat_completion(content: str):
    """Build a MagicMock shaped like an openai ChatCompletion response, so
    mod.llm_client.chat.completions.create(...) can be told to "reply" with
    a given raw JSON string.
    """
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


def test_health_check_reports_the_active_backend():
    with _client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "radar keyword service",
        "backend": "bert",
        "default_bert_model": "pubmedbert",
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
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion(
        '{"keywords": ["insulin", "  ", "glucose"]}'
    )

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


def test_extract_iris_via_ollama_backend_returns_a_match_per_extracted_keyword(load_backend, monkeypatch):
    mod = load_backend("ollama")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": ["insulin"]}')

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return {"iri": f"https://example.org/{keyword}"}

    monkeypatch.setattr(mod, "search_tib_best_match", fake_search)

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin"})

    assert response.status_code == 200
    assert response.json() == {"insulin": {"iri": "https://example.org/insulin"}}


def test_extract_iris_via_ollama_backend_requests_json_output(load_backend):
    mod = load_backend("ollama")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": []}')

    with _client(mod) as client:
        client.post("/extract-iris", json={"document": "insulin study"})

    _, kwargs = mod.llm_client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_extract_iris_via_ollama_backend_returns_500_when_extraction_fails(load_backend):
    mod = load_backend("ollama")

    def boom(*args, **kwargs):
        raise RuntimeError("llm exploded")

    mod.llm_client.chat.completions.create.side_effect = boom

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 500
    assert "llm exploded" in response.json()["detail"]


def test_extract_iris_via_ollama_backend_returns_500_when_the_reply_is_not_valid_json(load_backend):
    # response_format={"type": "json_object"} makes this vanishingly unlikely with a
    # real Ollama server, but the failure mode should still be an honest 500 rather
    # than an unrelated crash further down the pipeline if it ever happens.
    mod = load_backend("ollama")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion("not json")

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 500


def test_extract_iris_via_ollama_backend_handles_no_keywords_found(load_backend):
    mod = load_backend("ollama")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": []}')

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "..."})

    assert response.status_code == 200
    assert response.json() == {}
