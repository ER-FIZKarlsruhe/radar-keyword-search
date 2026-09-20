from fastapi.testclient import TestClient

import iri_api


def _client(mod=iri_api):
    return TestClient(mod.app)


def test_list_models_reports_the_full_registry_and_default():
    with _client() as client:
        response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "keybert"
    assert body["default_model"] == "pubmedbert"

    names = {m["name"] for m in body["models"]}
    assert names == set(iri_api.BERT_MODEL_REGISTRY)

    default_entry = next(m for m in body["models"] if m["name"] == "pubmedbert")
    assert default_entry["default"] is True
    assert default_entry["hf_model"] == iri_api.BERT_MODEL_REGISTRY["pubmedbert"]
    # The default model is loaded eagerly at startup.
    assert default_entry["loaded"] is True

    other_entry = next(m for m in body["models"] if m["name"] == "biobert")
    assert other_entry["default"] is False


def test_extract_iris_rejects_an_unknown_model():
    with _client() as client:
        response = client.post(
            "/extract-iris",
            json={"document": "insulin study", "model": "not-a-real-model"},
        )

    assert response.status_code == 400
    assert "Unknown BERT model" in response.json()["detail"]


def test_extract_iris_rejects_a_model_param_when_backend_is_keyllm(load_backend):
    mod = load_backend("keyllm")

    with _client(mod) as client:
        response = client.post(
            "/extract-iris",
            json={"document": "insulin study", "model": "biobert"},
        )

    assert response.status_code == 400
    assert "EXTRACTION_BACKEND=keybert" in response.json()["detail"]


def test_extract_iris_loads_a_non_default_model_only_once(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("insulin", 0.9)]
    calls_before = iri_api.KeyBERT.call_count

    async def fake_search(keyword, ontology, ontology_collection, threshold, client):
        return None

    monkeypatch.setattr(iri_api, "search_tib_best_match", fake_search)

    with _client() as client:
        for _ in range(2):
            response = client.post(
                "/extract-iris",
                json={"document": "insulin study", "model": "biobert"},
            )
            assert response.status_code == 200

    # One new KeyBERT instance for "biobert", cached and reused on the 2nd call.
    assert iri_api.KeyBERT.call_count == calls_before + 1
    assert "biobert" in iri_api._bert_model_cache


def test_invalid_default_bert_model_raises_at_import_time(load_backend):
    import pytest

    with pytest.raises(RuntimeError, match="Invalid BERT_MODEL"):
        load_backend("keybert", BERT_MODEL="not-a-real-model")
