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
        "backend": "keybert",
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


def test_extract_iris_via_keyllm_backend_filters_blank_keywords(load_backend, monkeypatch):
    mod = load_backend("keyllm")
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

    async def fake_resolve(ontology_collection, client):
        return ontology_collection.upper()

    monkeypatch.setattr(mod, "resolve_collection_ids", fake_resolve)

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
    # The collection is resolved to TIB's exact id once, before the per-keyword searches
    assert calls == [
        ("insulin", "chebi", "CS"),
        ("glucose", "chebi", "CS"),
    ]


def test_extract_iris_via_keyllm_backend_returns_a_match_per_extracted_keyword(load_backend, monkeypatch):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": ["insulin"]}')

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return {"iri": f"https://example.org/{keyword}"}

    monkeypatch.setattr(mod, "search_tib_best_match", fake_search)

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin"})

    assert response.status_code == 200
    assert response.json() == {"insulin": {"iri": "https://example.org/insulin"}}


def test_extract_iris_via_keyllm_backend_requests_json_output(load_backend):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": []}')

    with _client(mod) as client:
        client.post("/extract-iris", json={"document": "insulin study"})

    _, kwargs = mod.llm_client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_extract_iris_via_keyllm_backend_returns_500_when_extraction_fails(load_backend):
    mod = load_backend("keyllm")

    def boom(*args, **kwargs):
        raise RuntimeError("llm exploded")

    mod.llm_client.chat.completions.create.side_effect = boom

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 500
    assert "llm exploded" in response.json()["detail"]


def test_extract_iris_via_keyllm_backend_returns_500_when_the_reply_is_not_valid_json(load_backend):
    # response_format={"type": "json_object"} makes this vanishingly unlikely with a
    # real Ollama server, but the failure mode should still be an honest 500 rather
    # than an unrelated crash further down the pipeline if it ever happens.
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion("not json")

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "insulin study"})

    assert response.status_code == 500


def test_extract_iris_via_keyllm_backend_handles_no_keywords_found(load_backend):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": []}')

    with _client(mod) as client:
        response = client.post("/extract-iris", json={"document": "..."})

    assert response.status_code == 200
    assert response.json() == {}


def _fake_lookups(monkeypatch, calls, tib_result=None, gnd_result=None):
    async def fake_tib(keyword, ontology, ontology_collection, threshold, client):
        calls.append(("tib", keyword))
        return tib_result

    async def fake_gnd(keyword, threshold, client):
        calls.append(("gnd", keyword))
        return gnd_result

    async def fake_resolve(ontology_collection, client):
        calls.append(("resolve", ontology_collection))
        return ontology_collection

    monkeypatch.setattr(iri_api, "search_tib_best_match", fake_tib)
    monkeypatch.setattr(iri_api, "search_gnd_best_match", fake_gnd)
    monkeypatch.setattr(iri_api, "resolve_collection_ids", fake_resolve)


def test_extract_iris_defaults_to_tib_only(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("edition", 0.9)]
    calls = []
    _fake_lookups(monkeypatch, calls)

    with _client() as client:
        response = client.post("/extract-iris", json={"document": "edition"})

    assert response.status_code == 200
    assert ("tib", "edition") in calls
    assert not any(source == "gnd" for source, _ in calls)


def test_extract_iris_with_gnd_skips_tib_entirely(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("edition", 0.9)]
    gnd = {"iri": "https://d-nb.info/gnd/4132033-5", "source": "gnd", "matched": True, "distance": 0}
    calls = []
    _fake_lookups(monkeypatch, calls, gnd_result=gnd)

    with _client() as client:
        response = client.post(
            "/extract-iris",
            json={"document": "edition", "terminology": "gnd", "ontology_collection": "nfdi4culture"},
        )

    assert response.status_code == 200
    assert response.json() == {"edition": gnd}
    assert calls == [("gnd", "edition")]


def test_extract_iris_with_both_returns_the_better_match(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("edition", 0.9)]
    tib = {"iri": "https://example.org/edition", "source": "tib", "matched": False, "distance": 6}
    gnd = {"iri": "https://d-nb.info/gnd/4132033-5", "source": "gnd", "matched": True, "distance": 0}
    calls = []
    _fake_lookups(monkeypatch, calls, tib_result=tib, gnd_result=gnd)

    with _client() as client:
        response = client.post("/extract-iris", json={"document": "edition", "terminology": "both"})

    assert response.status_code == 200
    assert response.json() == {"edition": gnd}
    assert {c[0] for c in calls} == {"tib", "gnd", "resolve"}


def test_extract_iris_rejects_an_unknown_terminology():
    with _client() as client:
        response = client.post("/extract-iris", json={"document": "x", "terminology": "wikidata"})

    assert response.status_code == 422


def test_extract_iris_passes_max_distance_as_the_threshold(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("napoleon", 0.9)]
    thresholds = []

    async def fake_tib(keyword, ontology, ontology_collection, threshold, client):
        thresholds.append(("tib", threshold))

    async def fake_gnd(keyword, threshold, client):
        thresholds.append(("gnd", threshold))

    async def fake_resolve(ontology_collection, client):
        return ontology_collection

    monkeypatch.setattr(iri_api, "search_tib_best_match", fake_tib)
    monkeypatch.setattr(iri_api, "search_gnd_best_match", fake_gnd)
    monkeypatch.setattr(iri_api, "resolve_collection_ids", fake_resolve)

    with _client() as client:
        client.post("/extract-iris", json={"document": "x", "terminology": "both", "max_distance": 0})
        client.post("/extract-iris", json={"document": "x", "terminology": "both"})

    assert sorted(thresholds[:2]) == [("gnd", 0), ("tib", 0)]
    assert sorted(thresholds[2:]) == [("gnd", 5), ("tib", 5)]


def test_extract_iris_rejects_a_max_distance_above_the_threshold():
    with _client() as client:
        response = client.post("/extract-iris", json={"document": "x", "max_distance": 6})

    assert response.status_code == 422
