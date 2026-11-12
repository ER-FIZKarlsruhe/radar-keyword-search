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
#
# ---------------------------------------------------------------------------
# Portions of this file are derived from KeyBERT v0.9.0
# (https://github.com/MaartenGr/KeyBERT, keybert/_model.py), used under the
# MIT License:
#
#   Copyright (c) 2020, Maarten P. Grootendorst
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
# ---------------------------------------------------------------------------
"""Embedding-based keyword extraction: KeyBERT's algorithm, ported from KeyBERT.

Credit: the algorithm and the code this module is derived from are KeyBERT's,
by Maarten P. Grootendorst (https://github.com/MaartenGr/KeyBERT,
https://doi.org/10.5281/zenodo.4461265), used under the MIT License (see the
notice above and the NOTICE file). Please cite KeyBERT, not this module.

This replaces the `keybert` package, of which the service only ever used one
code path: KeyBERT.extract_keywords() on a single document with plain cosine
ranking (no MMR, no MaxSum, no seed keywords, no LLM). That path is:

1. Candidate words: scikit-learn's CountVectorizer over the document, with the
   requested n-gram range and stop-word list.
2. Embed the document and every candidate with the same embedding model.
3. Rank candidates by cosine similarity to the document embedding and keep the
   top_n, most similar first, scores rounded to 4 decimals.

Output is identical to keybert==0.9.0 for the same embedder and arguments
(see tests/test_keyword_extraction.py, whose expected values were produced by
keybert 0.9.0 itself).

Passing `candidates` (e.g. noun phrases from noun_phrases.py) skips step 1:
those candidates are ranked instead of the document's own words.
"""
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Takes a sequence of strings, returns a 2-D array: one embedding row per string.
EmbedFn = Callable[[Sequence[str]], np.ndarray]


class KeywordExtractor:
    """Rank a document's own words by how similar their embedding is to the document's."""

    def __init__(self, embed: EmbedFn):
        self._embed = embed

    def extract_keywords(
        self,
        doc: str,
        top_n: int = 5,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str], None] = "english",
        candidates: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return up to top_n (keyword, cosine score) pairs, most relevant first.

        candidates, if given, are ranked instead of the document's own words;
        keyphrase_ngram_range and stop_words then don't apply.
        """
        if not doc or top_n <= 0:
            return []

        if candidates is None:
            try:
                vectorizer = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words).fit([doc])
            except ValueError:
                # Raised when the document has no candidates left, e.g. only stop words.
                return []
            # Fitted on this single document, so every vocabulary entry occurs in it.
            candidates = vectorizer.get_feature_names_out()
        else:
            # Sorted like CountVectorizer's vocabulary, so ties are broken the same way.
            candidates = sorted(set(candidates))
            if not candidates:
                return []

        doc_embedding = np.asarray(self._embed([doc])).reshape(1, -1)
        candidate_embeddings = np.asarray(self._embed(candidates))

        similarities = cosine_similarity(doc_embedding, candidate_embeddings)[0]
        best_first = similarities.argsort()[-top_n:][::-1]
        return [(str(candidates[i]), round(float(similarities[i]), 4)) for i in best_first]
