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
    embedding.embed(["insulin"])

    docs = embedding._tokenizer.call_args.args[0]
    assert docs == ["query: insulin"]


def test_bert_embedding_hands_the_tokenizer_plain_strings_and_truncates():
    # KeywordExtractor embeds its candidate words as a numpy array of numpy str_ values.
    class FakeStr(str):
        pass

    embedding = iri_api._BertEmbedding("some/model")
    embedding.embed((FakeStr("insulin"), FakeStr("glucose")))

    call = embedding._tokenizer.call_args
    docs = call.args[0]
    assert docs == ["insulin", "glucose"]
    assert type(docs) is list and all(type(d) is str for d in docs)
    assert call.kwargs["truncation"] is True
    assert call.kwargs["max_length"] == 512


def test_keyword_extractor_embeds_with_the_registry_model():
    iri_api._bert_model_cache.pop("scibert", None)
    iri_api.get_bert_model("scibert")

    embed = iri_api.KeywordExtractor.call_args.args[0]
    assert isinstance(embed.__self__, iri_api._BertEmbedding)
    iri_api.AutoModel.from_pretrained.assert_called_with(iri_api.BERT_MODEL_REGISTRY["scibert"]["hf_model"])


def test_bert_embedding_loads_an_explicit_config_for_checkpoints_without_a_model_type():
    # Older checkpoints (e.g. deepset/gbert-base) don't name their architecture in
    # config.json, which transformers >= 5 needs to pick the tokenizer and model.
    iri_api.CONFIG_MAPPING.__getitem__.reset_mock()
    iri_api._BertEmbedding("deepset/gbert-base", model_type="bert")

    iri_api.CONFIG_MAPPING.__getitem__.assert_called_with("bert")
    config = iri_api.CONFIG_MAPPING["bert"].from_pretrained.return_value
    assert iri_api.AutoTokenizer.from_pretrained.call_args.kwargs == {"config": config}
    assert iri_api.AutoModel.from_pretrained.call_args.kwargs == {"config": config}


def test_bert_embedding_lets_transformers_read_the_config_by_default():
    iri_api._BertEmbedding("allenai/scibert_scivocab_uncased")

    assert iri_api.AutoTokenizer.from_pretrained.call_args.kwargs == {}
    assert iri_api.AutoModel.from_pretrained.call_args.kwargs == {}


def test_extract_iris_rejects_an_unknown_model():
    with _client() as client:
        response = client.post(
            "/extract-iris",
            json={"document": "insulin study", "model": "not-a-real-model"},
        )

    assert response.status_code == 400
    assert "Unknown BERT model" in response.json()["detail"]


def test_extract_iris_loads_a_non_default_model_only_once(monkeypatch):
    iri_api.kw_model.extract_keywords.return_value = [("insulin", 0.9)]
    calls_before = iri_api.KeywordExtractor.call_count

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

    # One new KeywordExtractor for "biobert", cached and reused on the 2nd call.
    assert iri_api.KeywordExtractor.call_count == calls_before + 1
    assert "biobert" in iri_api._bert_model_cache


def test_invalid_default_bert_model_raises_at_import_time(reload_iri_api):
    import pytest

    with pytest.raises(RuntimeError, match="Invalid BERT_MODEL"):
        reload_iri_api(BERT_MODEL="not-a-real-model")
