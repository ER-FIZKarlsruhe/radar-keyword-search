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
    assert default_entry["hf_model"] == iri_api.BERT_MODEL_REGISTRY["pubmedbert"]["hf_model"]
    assert default_entry["label"] == iri_api.BERT_MODEL_REGISTRY["pubmedbert"]["label"]
    # The default model is loaded eagerly at startup.
    assert default_entry["loaded"] is True

    other_entry = next(m for m in body["models"] if m["name"] == "biobert")
    assert other_entry["default"] is False


def test_list_models_reports_kind_languages_and_domains():
    with _client() as client:
        body = client.get("/models").json()

    e5 = next(m for m in body["models"] if m["name"] == "multilingual-e5")
    assert e5["kind"] == "sentence-embedding"
    assert e5["languages"] == ["multilingual"]
    assert "Interdisciplinary" in e5["domains"]


def test_every_registry_entry_is_well_formed():
    for name, entry in iri_api.BERT_MODEL_REGISTRY.items():
        assert entry["kind"] in {"sentence-embedding", "masked-lm"}, name
        assert entry["languages"], name
        assert entry["domains"], name


def test_every_domain_has_at_least_two_candidate_models():
    models_per_domain = {}
    for name, entry in iri_api.BERT_MODEL_REGISTRY.items():
        for domain in entry["domains"]:
            models_per_domain.setdefault(domain, set()).add(name)

    thin = {d: m for d, m in models_per_domain.items() if len(m) < 2}
    assert not thin


def test_bert_embedding_prepends_the_registry_prefix():
    embedding = iri_api._BertEmbedding("some/model", prefix="query: ")
    embedding(["insulin"])

    docs = embedding._tokenizer.call_args.args[0]
    assert docs == ["query: insulin"]


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
