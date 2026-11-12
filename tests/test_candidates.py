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

"""/extract-iris's `candidates` and `language` fields: which keyword candidates are ranked.

"words" (the default) must keep the service's behaviour exactly as before:
single words, no spaCy.
"""
import pytest
from fastapi.testclient import TestClient

import iri_api


@pytest.fixture(autouse=True)
def _no_lookups(monkeypatch):
    async def no_match(*args, **kwargs):
        return None

    monkeypatch.setattr(iri_api, "search_tib_best_match", no_match)
    monkeypatch.setattr(iri_api, "search_gnd_best_match", no_match)
    iri_api.kw_model.extract_keywords.return_value = []


def _post(payload):
    with TestClient(iri_api.app) as client:
        return client.post("/extract-iris", json={"document": "Insulin resistance in type 2 diabetes.", **payload})


def test_by_default_single_words_are_ranked_without_noun_phrases():
    response = _post({})

    assert response.status_code == 200
    _, kwargs = iri_api.kw_model.extract_keywords.call_args
    assert kwargs["keyphrase_ngram_range"] == (1, 1)
    assert "candidates" not in kwargs
    iri_api.noun_phrase_candidates.assert_not_called()


def test_words_are_the_explicit_default():
    _post({"candidates": "words"})

    _, kwargs = iri_api.kw_model.extract_keywords.call_args
    assert kwargs["keyphrase_ngram_range"] == (1, 1)
    assert "candidates" not in kwargs
    iri_api.noun_phrase_candidates.assert_not_called()


def test_ngrams_rank_word_sequences_of_up_to_three_words():
    _post({"candidates": "ngrams"})

    _, kwargs = iri_api.kw_model.extract_keywords.call_args
    assert kwargs["keyphrase_ngram_range"] == (1, 3)
    assert "candidates" not in kwargs
    iri_api.noun_phrase_candidates.assert_not_called()


def test_phrases_rank_the_documents_noun_phrases():
    iri_api.noun_phrase_candidates.return_value = ["insulin resistance", "type 2 diabetes"]
    iri_api.kw_model.extract_keywords.return_value = [("insulin resistance", 0.9)]

    response = _post({"candidates": "phrases"})

    assert response.status_code == 200
    assert response.json() == {"insulin resistance": None}
    iri_api.noun_phrase_candidates.assert_called_once_with("Insulin resistance in type 2 diabetes.", None)
    _, kwargs = iri_api.kw_model.extract_keywords.call_args
    assert kwargs["candidates"] == ["insulin resistance", "type 2 diabetes"]


def test_phrases_pass_a_requested_language_on():
    _post({"candidates": "phrases", "language": "de"})

    iri_api.noun_phrase_candidates.assert_called_once_with("Insulin resistance in type 2 diabetes.", "de")


def test_language_is_ignored_for_words():
    response = _post({"language": "de"})

    assert response.status_code == 200
    iri_api.noun_phrase_candidates.assert_not_called()


@pytest.mark.parametrize("payload", [{"candidates": "sentences"}, {"language": "fr"}])
def test_unknown_candidates_or_languages_are_rejected(payload):
    response = _post(payload)

    assert response.status_code == 422
    iri_api.kw_model.extract_keywords.assert_not_called()
