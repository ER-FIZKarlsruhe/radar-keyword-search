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

"""Stop words and the length-aware match threshold.

The cases are real ones from the 2026-11-08 evaluation against the online
service (evaluation/results/*_20261108-12*.json).
"""
import httpx
import pytest

import iri_api
from iri_api import allowed_distance, search_gnd_best_match, search_tib_best_match
from stop_words import GERMAN_STOP_WORDS, STOP_WORDS


# -------------------------------
# Stop words
# -------------------------------
@pytest.mark.parametrize("word", ["hier", "und", "der", "nicht", "für", "dass", "fuer", "the", "and"])
def test_german_and_english_filler_words_are_stop_words(word):
    assert word in STOP_WORDS


def test_war_stays_a_keyword_candidate():
    # German "war" (= was) is a Snowball stop word, but "war" is a key topic in history datasets.
    assert "war" not in STOP_WORDS
    assert "war" not in GERMAN_STOP_WORDS


def test_stop_words_are_a_sorted_list_without_duplicates():
    assert STOP_WORDS == sorted(set(STOP_WORDS))


def test_keyword_extraction_uses_the_english_and_german_stop_words():
    iri_api.kw_model.extract_keywords.return_value = []

    iri_api._extract_keyword_list("Hier wird die Edition beschrieben.")

    assert iri_api.kw_model.extract_keywords.call_args.kwargs["stop_words"] is STOP_WORDS


# -------------------------------
# Length-aware threshold
# -------------------------------
@pytest.mark.parametrize(
    "keyword, expected",
    [("ké", 0), ("hier", 1), ("witches", 1), ("heinrich", 2), ("nanoparticles", 3),
     ("streichinstrumenten", 4), ("forschungsdatenmanagement", 5)],
)
def test_allowed_distance_is_a_quarter_of_the_keyword_length_capped_at_5(keyword, expected):
    assert allowed_distance(keyword, 5) == expected


def test_allowed_distance_never_exceeds_the_requested_max_distance():
    assert allowed_distance("forschungsdatenmanagement", 2) == 2
    assert allowed_distance("forschungsdatenmanagement", 0) == 0


def _tib(label):
    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(200)
        return httpx.Response(200, json={"response": {"docs": [{"iri": "https://example.org/t", "label": label}]}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _gnd(name):
    member = {"id": "https://d-nb.info/gnd/1", "gndIdentifier": "1", "preferredName": name, "type": ["SubjectHeading"]}

    def handler(request):
        return httpx.Response(200, json={"member": [member]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "keyword, label",
    [
        ("hier", "usko"),               # a 4-letter keyword used to match any 4-letter term
        ("witches", "Witches game"),    # keyword + extra word
        ("beethoven", "Beethoven Duo"),
        ("results", "results in"),
    ],
)
async def test_tib_rejects_matches_that_differ_too_much_for_the_keyword_length(keyword, label):
    async with _tib(label) as client:
        result = await search_tib_best_match(keyword, None, None, threshold=5, client=client)

    assert result["best_term"] == label
    assert result["distance"] <= 5  # within the old fixed threshold ...
    assert result["matched"] is False  # ... but no longer accepted


@pytest.mark.parametrize(
    "keyword, label",
    [("nanoparticles", "nanoparticle"), ("sulfonamides", "sulfonamide"), ("oxides", "oxide")],
)
async def test_tib_still_accepts_small_edits(keyword, label):
    async with _tib(label) as client:
        result = await search_tib_best_match(keyword, None, None, threshold=5, client=client)

    assert result["distance"] == 1
    assert result["matched"] is True


async def test_gnd_applies_the_same_rule():
    async with _gnd("Frühe Gene") as client:
        prefix = await search_gnd_best_match("frühe", threshold=5, client=client)
    async with _gnd("Glück") as client:
        umlaut = await search_gnd_best_match("gluck", threshold=5, client=client)

    assert prefix["matched"] is False
    assert umlaut["distance"] == 1 and umlaut["matched"] is True
