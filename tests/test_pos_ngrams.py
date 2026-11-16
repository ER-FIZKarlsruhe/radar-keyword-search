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

"""Tests for pos_ngrams.py: phrase candidates and their ranking (candidates="pos-ngrams").

conftest.py replaces the pos_ngrams module with a MagicMock so that iri_api's tests
don't need spaCy; here the real module is loaded from its file under a private name
instead, with the real spaCy models from requirements-test.txt. The BERT model is
replaced by a bag-of-words embedding.
"""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_pos_ngrams_real", Path(__file__).resolve().parent.parent / "pos_ngrams.py"
)
pos_ngrams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pos_ngrams)


@pytest.fixture(scope="module")
def extractor():
    # Loading the spaCy pipelines takes a few seconds: share one instance per module.
    return pos_ngrams.PosNgramExtractor()


def _bag_of_words(vocabulary):
    """An embed function: one dimension per vocabulary word, plus one so no vector is zero."""
    index = {w: i for i, w in enumerate(vocabulary)}

    def embed(texts):
        vectors = np.zeros((len(texts), len(vocabulary) + 1))
        vectors[:, -1] = 0.1
        for row, text in enumerate(texts):
            for word in re.findall(r"[\w-]+", str(text).lower()):
                if word in index:
                    vectors[row, index[word]] += 1
        return vectors

    return embed


# -------------------------------
# Language of a paragraph
# -------------------------------
def test_detects_german_and_english_by_their_stop_words():
    assert pos_ngrams.detect_language("Die Briefe Goethes werden im Archiv aufbewahrt und ediert.") == "de"
    assert pos_ngrams.detect_language("The letters of Goethe are kept in the archive and edited.") == "en"
    assert pos_ngrams.detect_language("Forschungsdatenmanagement Schauspielmusik") is None


def test_each_paragraph_gets_its_own_language():
    text = "Die Briefe Goethes werden im Archiv aufbewahrt.\n\nThe letters of Goethe are kept in the archive."

    assert pos_ngrams.split_by_language(text) == [
        ("de", "Die Briefe Goethes werden im Archiv aufbewahrt."),
        ("en", "The letters of Goethe are kept in the archive."),
    ]


def test_a_paragraph_without_stop_words_takes_the_documents_language():
    text = "Schauspielmusik\n\nDie Musik wird im Theater gespielt und ist für das Stück geschrieben."

    assert [lang for lang, _ in pos_ngrams.split_by_language(text)] == ["de", "de"]


# -------------------------------
# Candidates
# -------------------------------
def test_candidates_are_noun_sequences_of_up_to_three_words(extractor):
    candidates = extractor.candidates("Keratosis pilaris is a common skin disease. It causes rough patches.")

    assert candidates == ["disease", "keratosis", "keratosis pilaris", "patches", "pilaris",
                          "rough patches", "skin", "skin disease"]


def test_generic_adjectives_are_left_out_also_when_inflected(extractor):
    assert "optical activity" in extractor.candidates("Strong optical activity was observed.")
    assert "strong optical activity" not in extractor.candidates("Strong optical activity was observed.")
    assert extractor.candidates("Die verschiedenen Methoden der Edition wurden verglichen.") == ["edition", "methoden"]


def test_hyphenated_words_are_one_word(extractor):
    assert "historisch-kritische edition" in extractor.candidates("Die historisch-kritische Edition der Briefe erschien.")
    assert "anti-inflammatory drug" in extractor.candidates("The anti-inflammatory drug reduced pain in children.")


def test_candidates_never_span_sentences(extractor):
    candidates = extractor.candidates("Wir untersuchen die Schauspielmusik. Musik und Theater.")

    assert candidates == ["musik", "schauspielmusik", "theater"]


def test_a_text_without_nouns_has_no_candidates(extractor):
    assert extractor.candidates("") == []
    assert extractor.candidates("und oder aber") == []


# -------------------------------
# Ranking
# -------------------------------
def test_returns_at_most_top_n_distinct_keywords(extractor):
    text = "Keratosis pilaris is a common skin disease. It causes rough patches on the skin."
    embed = _bag_of_words(["keratosis", "pilaris", "skin", "disease", "patches", "rough"])

    keywords = extractor(text, embed, top_n=3)

    assert len(keywords) == 3
    assert len(set(keywords)) == 3
    assert set(keywords) <= set(extractor.candidates(text))


def test_inflected_variants_are_picked_only_once(extractor):
    text = "The patient was examined. Each patient and all patients were examined again."
    embed = _bag_of_words(["patient", "patients"])

    keywords = extractor(text, embed, top_n=10)

    assert len([k for k in keywords if k in ("patient", "patients")]) == 1


def test_frequent_early_phrases_rank_first(extractor):
    # Every candidate is equally similar to the document here, so position and frequency decide.
    text = "Schauspielmusik und Schauspielmusik. Später kommt auch das Archiv vor."
    embed = _bag_of_words([])

    assert extractor(text, embed, top_n=1) == ["schauspielmusik"]


def test_nothing_to_rank_returns_an_empty_list(extractor):
    embed = _bag_of_words(["skin"])

    assert extractor("und oder aber", embed) == []
    assert extractor("Skin disease.", embed, top_n=0) == []


def test_is_variant_allows_a_quarter_of_the_word_length():
    assert pos_ngrams._is_variant("keratose", "keratosen")
    assert pos_ngrams._is_variant("skin disease", "skin diseases")
    assert not pos_ngrams._is_variant("skin", "skin disease")
    assert not pos_ngrams._is_variant("cat", "car")
