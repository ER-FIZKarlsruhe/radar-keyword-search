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

"""Tests for the in-house KeyBERT replacement in keyword_extraction.py.

conftest.py replaces the keyword_extraction module with a MagicMock so that
iri_api's tests don't need numpy/scikit-learn; here the real module is loaded
from its file under a private name instead.

The expected values below were produced by keybert==0.9.0's
KeyBERT.extract_keywords(doc, keyphrase_ngram_range=(1, 1),
stop_words='english', top_n=...) using the same fake embedder, so these tests
pin the replacement to KeyBERT's exact behaviour: same candidate words, same
ranking, same rounded scores.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_keyword_extraction_real", Path(__file__).resolve().parent.parent / "keyword_extraction.py"
)
keyword_extraction = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(keyword_extraction)


def fake_embed(texts):
    """Deterministic stand-in for a BERT model: sum of per-token hash vectors."""
    rows = []
    for text in texts:
        vector = np.zeros(16)
        for token in str(text).lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            vector += np.frombuffer(digest[:16], dtype=np.uint8).astype(float) - 127.5
        rows.append(vector)
    return np.vstack(rows)


@pytest.fixture
def extractor():
    return keyword_extraction.KeywordExtractor(fake_embed)


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            "Insulin resistance and glucose metabolism in patients with type 2 diabetes mellitus.",
            [("resistance", 0.6494), ("metabolism", 0.6426), ("insulin", 0.3582), ("type", 0.305), ("diabetes", 0.2442)],
        ),
        (
            "The edition of Goethe's letters, compiled from manuscripts held in Weimar archives, documents correspondence.",
            [("documents", 0.7282), ("manuscripts", 0.3669), ("held", 0.3477), ("weimar", 0.3186), ("compiled", 0.2842)],
        ),
        (
            "Protein folding protein misfolding amyloid amyloid plaques Alzheimer disease neurons",
            [("protein", 0.724), ("amyloid", 0.6702), ("disease", 0.4238), ("neurons", 0.4076), ("alzheimer", 0.3521)],
        ),
        ("Graphene graphene graphene", [("graphene", 1.0)]),
    ],
)
def test_matches_keybert_0_9_0(extractor, doc, expected):
    assert extractor.extract_keywords(doc, top_n=5) == expected


def test_top_n_limits_the_result_and_keeps_the_best_first(extractor):
    doc = "Insulin resistance and glucose metabolism in patients with type 2 diabetes mellitus."
    assert extractor.extract_keywords(doc, top_n=2) == [("resistance", 0.6494), ("metabolism", 0.6426)]


def test_top_n_larger_than_the_candidate_count_returns_every_candidate(extractor):
    keywords = extractor.extract_keywords("insulin glucose", top_n=50)
    assert sorted(kw for kw, _ in keywords) == ["glucose", "insulin"]


@pytest.mark.parametrize("doc", ["", "the and of a"])
def test_documents_without_candidates_yield_nothing(extractor, doc):
    assert extractor.extract_keywords(doc, top_n=5) == []


def test_non_positive_top_n_yields_nothing(extractor):
    assert extractor.extract_keywords("insulin glucose", top_n=0) == []


def test_results_are_plain_str_and_float(extractor):
    keyword, score = extractor.extract_keywords("insulin glucose", top_n=1)[0]
    assert type(keyword) is str and type(score) is float
