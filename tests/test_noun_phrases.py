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

"""Tests for noun_phrases.py: keyword candidates from spaCy noun chunks.

conftest.py replaces the noun_phrases module with a MagicMock so that iri_api's
tests don't need spaCy; here the real module is loaded from its file under a
private name instead, with the real spaCy models from requirements-test.txt.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_noun_phrases_real", Path(__file__).resolve().parent.parent / "noun_phrases.py"
)
noun_phrases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(noun_phrases)


@pytest.fixture(scope="module")
def candidates():
    # Loading the spaCy pipelines takes a few seconds: share one instance per module.
    return noun_phrases.NounPhraseCandidates()


# -------------------------------
# Language of a paragraph
# -------------------------------
def test_detects_german_by_its_stop_words():
    assert noun_phrases.detect_language("Die Briefe Goethes werden im Archiv aufbewahrt und ediert.") == "de"


def test_detects_english_by_its_stop_words():
    assert noun_phrases.detect_language("The letters of Goethe are kept in the archive and edited.") == "en"


def test_a_text_without_stop_words_has_no_detectable_language():
    assert noun_phrases.detect_language("Forschungsdatenmanagement Schauspielmusik") is None


def test_each_paragraph_gets_its_own_language():
    text = "Die Briefe Goethes werden im Archiv aufbewahrt.\n\nThe letters of Goethe are kept in the archive."

    assert noun_phrases.split_by_language(text) == [
        ("de", "Die Briefe Goethes werden im Archiv aufbewahrt."),
        ("en", "The letters of Goethe are kept in the archive."),
    ]


def test_a_paragraph_without_stop_words_takes_the_documents_language():
    # A title like this one has no stop words; the abstract below decides for it.
    text = "Historisch-kritische Edition\n\nDie Briefe Goethes werden im Archiv aufbewahrt und ediert."

    assert [lang for lang, _ in noun_phrases.split_by_language(text)] == ["de", "de"]


def test_a_document_without_any_stop_words_is_parsed_as_english():
    assert noun_phrases.split_by_language("Graphene nanoribbons") == [("en", "Graphene nanoribbons")]


def test_a_requested_language_applies_to_every_paragraph():
    text = "Die Briefe Goethes werden im Archiv aufbewahrt.\n\nThe letters of Goethe are kept in the archive."

    assert [lang for lang, _ in noun_phrases.split_by_language(text, "de")] == ["de", "de"]


def test_blank_paragraphs_are_skipped():
    assert noun_phrases.split_by_language("\n\n  \n\nThe archive of Weimar.\n\n") == [("en", "The archive of Weimar.")]


# -------------------------------
# Candidates
# -------------------------------
def test_keeps_multi_word_terms_whole(candidates):
    result = candidates("Insulin resistance is common in patients with type 2 diabetes mellitus.")

    assert "insulin resistance" in result
    assert "type 2 diabetes mellitus" in result


def test_keeps_german_adjective_noun_phrases_whole(candidates):
    result = candidates("Eine historisch-kritische Edition der Briefe aus dem Weimarer Archiv.")

    assert "historisch-kritische edition" in result
    assert "weimarer archiv" in result


def test_also_offers_the_single_nouns_of_a_phrase(candidates):
    result = candidates("Eine historisch-kritische Edition der Briefe aus dem Weimarer Archiv.")

    assert {"edition", "briefe", "archiv"} <= set(result)


def test_strips_articles_and_other_function_words_at_the_edges(candidates):
    result = candidates("The mitochondria is the powerhouse of the cell.\n\nDie Keratinozyten verstopfen die Haarfollikel.")

    assert {"mitochondria", "powerhouse", "cell"} <= set(result)
    assert not [c for c in result if c.split()[0] in {"the", "die", "dem", "der", "a", "eine"}]


def test_never_offers_verbs_or_stop_words(candidates):
    result = candidates("The mitochondria regulates apoptosis and we observed it.")

    assert "regulates" not in result
    assert "observed" not in result
    assert "we" not in result


def test_cuts_long_phrases_down_to_their_head(candidates):
    result = candidates("We restored the large old red wooden farm house door.")

    assert all(len(c.split()) <= noun_phrases.MAX_PHRASE_TOKENS for c in result)
    assert "wooden farm house door" in result


def test_drops_terms_with_unbalanced_brackets(candidates):
    # spaCy splits "Zn(II)" and "poly(lactide)" after the opening bracket's content.
    result = candidates("O^N^O-donor Zn(II) complexes for the depolymerization of poly(lactide).")

    assert not [c for c in result if c.count("(") != c.count(")")]
    assert "depolymerization" in result


def test_candidates_are_lowercase_unique_and_sorted(candidates):
    result = candidates("Edition und Edition. The edition of the Edition.")

    assert result == sorted(set(result))
    assert all(c == c.lower() for c in result)
    assert result.count("edition") == 1


@pytest.mark.parametrize("doc", ["", "   ", "the and of a", "und der die"])
def test_documents_without_nouns_yield_nothing(candidates, doc):
    assert candidates(doc) == []


def test_a_requested_language_selects_the_pipeline(candidates, monkeypatch):
    used = []
    real_pipeline = candidates._pipeline

    def tracking_pipeline(language):
        used.append(language)
        return real_pipeline(language)

    monkeypatch.setattr(candidates, "_pipeline", tracking_pipeline)

    candidates("The letters of Goethe are kept in the archive.", "de")

    assert used == ["de"]
