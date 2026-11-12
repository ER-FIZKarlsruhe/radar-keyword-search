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

"""Keyword candidates from spaCy noun phrases, for KeywordExtractor's `candidates`.

KeywordExtractor's own candidates are single words, so a term like "type 2
diabetes mellitus" or "historisch-kritische Edition" is torn apart before it is
ranked. This module offers the document's noun phrases (spaCy's noun_chunks)
and their single nouns instead:

1. Split the document into paragraphs and pick each one's language (German or
   English) by counting its stop words; a paragraph without any, typically a
   title, takes the language of the whole document.
2. Parse each paragraph with that language's spaCy pipeline.
3. Take every noun chunk, strip articles, prepositions etc. from its edges and
   keep at most its last MAX_PHRASE_TOKENS tokens (the head noun and its
   closest modifiers); take every noun and proper noun on its own as well.
4. Lowercase and deduplicate, like CountVectorizer's candidates.
"""
import re
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Span

from stop_words import ENGLISH_STOP_WORDS, GERMAN_STOP_WORDS, STOP_WORDS

# Pinned in requirements.txt; MIT licensed (see NOTICE).
SPACY_MODELS = {"de": "de_core_news_md", "en": "en_core_web_md"}

# Only the components noun_chunks and part-of-speech tags need.
_EXCLUDED_COMPONENTS = ["ner", "lemmatizer"]

# Longer noun chunks are cut down to their last tokens: in both languages a chunk
# ends with its head noun, so this keeps the head and its closest modifiers
# ("the large old red wooden farm house door" -> "wooden farm house door").
MAX_PHRASE_TOKENS = 4

# Parts of speech a candidate must neither start nor end with.
_EDGE_POS = {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "PUNCT", "SCONJ", "SPACE", "SYM", "VERB", "X"}

# Single tokens offered as candidates on their own.
_NOUN_POS = {"NOUN", "PROPN"}

# Words that only occur in one language's stop word list decide a paragraph's language.
_GERMAN_ONLY = GERMAN_STOP_WORDS - ENGLISH_STOP_WORDS
_ENGLISH_ONLY = ENGLISH_STOP_WORDS - GERMAN_STOP_WORDS

# Language of a document in which no paragraph has stop words.
_FALLBACK_LANGUAGE = "en"

_STOP_WORDS = frozenset(STOP_WORDS)


def detect_language(text: str) -> Optional[str]:
    """"de" or "en", whichever has more stop words in text; None if neither has any or it's a tie."""
    words = re.findall(r"\w+", text.lower())
    german = sum(1 for w in words if w in _GERMAN_ONLY)
    english = sum(1 for w in words if w in _ENGLISH_ONLY)
    if german == english:
        return None
    return "de" if german > english else "en"


def split_by_language(text: str, language: Optional[str] = None) -> List[Tuple[str, str]]:
    """Split text into (language, paragraph) pairs; language, if given, applies to every paragraph."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if language:
        return [(language, p) for p in paragraphs]
    fallback = detect_language(text) or _FALLBACK_LANGUAGE
    return [(detect_language(p) or fallback, p) for p in paragraphs]


def _is_edge_token(token) -> bool:
    return token.pos_ in _EDGE_POS or token.lower_ in _STOP_WORDS or not any(c.isalnum() for c in token.text)


def _trim(span: Span) -> Optional[Span]:
    """span without function words at its edges, cut to its last MAX_PHRASE_TOKENS tokens."""
    start, end = span.start, span.end
    for _ in range(2):  # Cutting the length can uncover a new edge token on the left.
        while start < end and _is_edge_token(span.doc[start]):
            start += 1
        while end > start and _is_edge_token(span.doc[end - 1]):
            end -= 1
        start = max(start, end - MAX_PHRASE_TOKENS)
    return span.doc[start:end] if start < end else None


def _normalize(span: Span) -> Optional[str]:
    """The candidate text of span, or None if it isn't usable as one."""
    text = re.sub(r"\s+", " ", span.text).strip().lower()
    if len(text) < 2 or text in _STOP_WORDS:
        return None
    # spaCy splits terms like "Zn(II)" after their opening bracket's content; such a
    # torn term would only produce misleading terminology matches.
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        return None
    return text


class NounPhraseCandidates:
    """Callable: document -> sorted, unique, lowercase noun-phrase candidates.

    The spaCy pipelines are loaded on first use. Not thread-safe; iri_api calls it
    under its extraction lock.
    """

    def __init__(self):
        self._pipelines: Dict[str, Language] = {}

    def _pipeline(self, language: str) -> Language:
        if language not in self._pipelines:
            self._pipelines[language] = spacy.load(SPACY_MODELS[language], exclude=_EXCLUDED_COMPONENTS)
        return self._pipelines[language]

    def __call__(self, doc: str, language: Optional[str] = None) -> List[str]:
        """Noun-phrase candidates of doc; language ("de"/"en"), if given, overrides detection."""
        found = set()
        for paragraph_language, paragraph in split_by_language(doc, language):
            parsed = self._pipeline(paragraph_language)(paragraph)
            spans = list(parsed.noun_chunks)
            spans += [parsed[t.i:t.i + 1] for t in parsed if t.pos_ in _NOUN_POS]
            for span in spans:
                trimmed = _trim(span)
                text = _normalize(trimmed) if trimmed is not None else None
                if text:
                    found.add(text)
        return sorted(found)
