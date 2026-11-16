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

"""Keywords of up to three words (`candidates: pos-ngrams`).

KeywordExtractor ranks single words, so a term like "keratosis pilaris" or
"follikuläre Hyperkeratose" is torn apart. This module extracts short phrases
instead and ranks them closer to how a person (or ChatGPT) picks keywords.

Candidates:

1. Split the document into paragraphs and pick each one's language (German or
   English) by counting its stop words; a paragraph without any, typically a
   title, takes the language of the whole document.
2. Tag each paragraph with that language's spaCy pipeline (parts of speech only).
3. Take every contiguous sequence of 1-3 words that contains a noun and never
   crosses a sentence end, punctuation, a stop word, a verb or a generic
   adjective ("new", "strong", "verschiedene"); it must not start or end with a
   number. Words joined by a hyphen ("co-composting") count as one word.

Ranking: each candidate's cosine similarity to the document (as KeywordExtractor
computes it) is standardised within the document and adjusted by

    score = z(sim) + LENGTH_BONUS[words] + W_POSITION * (1 - first_position) + W_FREQUENCY * ln(occurrences)

Keywords are then picked greedily, best score first: a candidate is skipped if it
is an inflected variant or a part of a picked keyword ("keratosis" after "keratosis
pilaris"), and penalised by MMR_LAMBDA times its largest similarity to a picked
keyword, so near-duplicates don't fill the list. A phrase that extends a picked
keyword is kept: it is often the more specific term ("acute respiratory failure").

The weights were tuned on the keyword benchmark (evaluation/benchmark/) against
ChatGPT's keywords for 243 abstracts (chemistry, medicine, humanities; German and
English) with the multilingual-e5 model: on held-out documents F1@10 rose from
0.21 (single words: 0.14) to 0.23. Skipping parts of picked keywords removes about
two thirds of the overlapping keywords at the same F1 vs. ChatGPT, but lowers F1 against
the reference keywords (GND subjects, author keywords) from 0.115 to 0.091.
"""
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import spacy
from spacy.language import Language

from stop_words import ENGLISH_STOP_WORDS, GERMAN_STOP_WORDS, STOP_WORDS

EmbedFn = Callable[[Sequence[str]], np.ndarray]

# Pinned in requirements.txt; MIT licensed (see NOTICE).
SPACY_MODELS = {"de": "de_core_news_md", "en": "en_core_web_md"}

# Only the tagger is needed.
_EXCLUDED_COMPONENTS = ["ner", "lemmatizer"]

MAX_WORDS = 3

# Tuned weights (see module docstring). Neighbouring values scored within 0.005.
LENGTH_BONUS = {1: 0.0, 2: 0.0, 3: -0.5}
W_POSITION = 0.5
W_FREQUENCY = 0.5
MMR_LAMBDA = 3.0

_STOP = {w.lower() for w in STOP_WORDS}
_KEEP_POS = {"NOUN", "PROPN", "ADJ", "X", "NUM"}
_NOUN_POS = {"NOUN", "PROPN", "X"}

# Words that only occur in one language's stop word list decide a paragraph's language.
_GERMAN_ONLY = GERMAN_STOP_WORDS - ENGLISH_STOP_WORDS
_ENGLISH_ONLY = ENGLISH_STOP_WORDS - GERMAN_STOP_WORDS

# Language of a document in which no paragraph has stop words.
_FALLBACK_LANGUAGE = "en"

# Adjectives that grade, count or relate rather than classify: never part of a keyword.
GENERIC_ADJECTIVES = {
    # English
    "new", "novel", "different", "various", "several", "many", "much", "more", "most", "other",
    "further", "strong", "weak", "high", "low", "large", "small", "big", "great", "important",
    "significant", "main", "major", "minor", "key", "specific", "possible", "potential",
    "current", "recent", "previous", "present", "first", "second", "third", "last", "final",
    "good", "better", "best", "distinct", "common", "general", "overall", "total", "simple",
    "effective", "efficient", "relevant", "certain", "particular", "similar", "same", "whole",
    "early", "late", "long", "short", "wide", "broad", "single", "multiple", "additional",
    "available", "useful", "interesting", "promising", "excellent", "remarkable", "typical",
    "increased", "reduced", "higher", "lower", "larger", "smaller", "greater", "due",
    "harmless", "excessive",
    # German (stems, see _is_generic_adjective)
    "neu", "verschieden", "unterschiedlich", "mehrere", "viel", "weit", "ander", "weiter",
    "stark", "schwach", "hoch", "niedrig", "groß", "klein", "wichtig", "signifikant",
    "wesentlich", "möglich", "potenziell", "aktuell", "bisherig", "erst", "zweit", "dritt",
    "letzt", "gut", "besser", "deutlich", "allgemein", "gesamt", "einfach", "effektiv",
    "relevant", "bestimmt", "besonder", "ähnlich", "gleich", "ganz", "früh", "spät", "lang",
    "kurz", "breit", "einzeln", "zusätzlich", "verfügbar", "nützlich", "interessant",
    "typisch", "häufig", "selten", "ausgeprägt", "erhöht", "vermindert", "zahlreich",
    "jeweilig", "folgend", "entsprechend", "sogenannt", "ungefährlich", "harmlos", "überschüssig",
}

# The pipelines run without lemmatizer, so German adjective endings are cut off by hand
# before looking a word up in GENERIC_ADJECTIVES ("kleinen" -> "klein").
_GERMAN_ENDINGS = ("eren", "erem", "erer", "eres", "en", "em", "er", "es", "e")

# A unit that ends a run of candidate words; also inserted at each sentence start.
_BREAK = ("", "BREAK")


def detect_language(text: str) -> Optional[str]:
    """"de" or "en", whichever has more stop words in text; None if neither has any or it's a tie."""
    words = re.findall(r"\w+", text.lower())
    german = sum(1 for w in words if w in _GERMAN_ONLY)
    english = sum(1 for w in words if w in _ENGLISH_ONLY)
    if german == english:
        return None
    return "de" if german > english else "en"


def split_by_language(text: str) -> List[Tuple[str, str]]:
    """Split text into (language, paragraph) pairs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    fallback = detect_language(text) or _FALLBACK_LANGUAGE
    return [(detect_language(p) or fallback, p) for p in paragraphs]


def _is_generic_adjective(word: str) -> bool:
    return word in GENERIC_ADJECTIVES or any(
        word.endswith(e) and word[:-len(e)] in GENERIC_ADJECTIVES for e in _GERMAN_ENDINGS)


def _units(doc) -> List[Tuple[str, str]]:
    """(text, part of speech) per word; words joined by a hyphen without spaces become one
    unit ("co-composting", "2-propanol") with the part of speech of the last part."""
    units: List[Tuple[str, str]] = []
    glue = False
    for t in doc:
        if glue:
            units[-1] = (units[-1][0] + t.text, t.pos_)
            glue = False
            continue
        if (t.text == "-" and units and units[-1] != _BREAK and t.i + 1 < len(doc)
                and not doc[t.i - 1].whitespace_ and not t.whitespace_):
            units[-1] = (units[-1][0] + "-", units[-1][1])
            glue = True
            continue
        if t.is_sent_start and units:
            units.append(_BREAK)  # candidates never span sentences
        units.append((t.text, t.pos_))
    return units


def _allowed(unit: Tuple[str, str]) -> bool:
    text, pos = unit
    low = text.lower()
    return (pos in _KEEP_POS and any(c.isalpha() for c in text) and len(low) > 1
            and "." not in low and not low.endswith("-") and low not in _STOP
            and not (pos == "ADJ" and _is_generic_adjective(low)))


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_variant(a: str, b: str) -> bool:
    """Same words up to inflection: equal word count, each pair within a quarter of its length."""
    wa, wb = a.split(), b.split()
    return len(wa) == len(wb) and all(_edit_distance(x, y) <= min(len(x), len(y)) // 4 for x, y in zip(wa, wb))


def _is_part_of(part: str, phrase: str) -> bool:
    """Every word of part occurs, up to inflection, in the longer phrase."""
    wp, wf = part.split(), phrase.split()
    return len(wp) < len(wf) and all(
        any(_edit_distance(x, y) <= min(len(x), len(y)) // 4 for y in wf) for x in wp)


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class PosNgramExtractor:
    """Callable: (document, embed function, top_n) -> keywords, best first.

    The spaCy pipelines are loaded on first use. Not thread-safe; iri_api calls it
    under its extraction lock.
    """

    def __init__(self):
        self._pipelines: Dict[str, Language] = {}

    def _pipeline(self, language: str) -> Language:
        if language not in self._pipelines:
            self._pipelines[language] = spacy.load(SPACY_MODELS[language], exclude=_EXCLUDED_COMPONENTS)
        return self._pipelines[language]

    def candidates(self, text: str) -> List[str]:
        """Sorted, unique, lowercase candidates (see module docstring)."""
        found = set()
        for language, paragraph in split_by_language(text):
            run: List[Tuple[str, str]] = []
            for unit in _units(self._pipeline(language)(paragraph)) + [_BREAK]:
                if _allowed(unit):
                    run.append(unit)
                    continue
                for i in range(len(run)):
                    for j in range(i + 1, min(i + MAX_WORDS, len(run)) + 1):
                        gram = run[i:j]
                        if any(p in _NOUN_POS for _, p in gram) and gram[0][1] != "NUM" and gram[-1][1] != "NUM":
                            found.add(" ".join(t.lower() for t, _ in gram))
                run = []
        return sorted(found)

    def __call__(self, text: str, embed: EmbedFn, top_n: int = 10) -> List[str]:
        candidates = self.candidates(text)
        if not candidates or top_n <= 0:
            return []
        vectors = _normalized(np.asarray(embed(candidates)))
        sim = vectors @ _normalized(np.asarray(embed([text]))[0])

        low = " ".join(text.lower().split())
        first, occurrences = [], []
        for c in candidates:
            hits = [m.start() for m in re.finditer(r"(?<!\w)" + re.escape(c) + r"(?!\w)", low)]
            first.append(hits[0] / max(len(low), 1) if hits else 0.5)
            occurrences.append(max(len(hits), 1))

        # e5-style models compress similarities into a narrow band: compare within the document.
        z = (sim - sim.mean()) / (sim.std() or 1.0)
        length = np.array([LENGTH_BONUS[min(len(c.split()), MAX_WORDS)] for c in candidates])
        score = (z + length + W_POSITION * (1 - np.array(first))
                 + W_FREQUENCY * np.log(np.array(occurrences, dtype=float)))

        chosen: List[int] = []
        remaining = list(np.argsort(-score, kind="stable"))
        while remaining and len(chosen) < top_n:
            if chosen:
                penalty = (vectors[remaining] @ vectors[chosen].T).max(1)
                pick = remaining[int(np.argmax(score[remaining] - MMR_LAMBDA * penalty))]
            else:
                pick = remaining[0]
            remaining.remove(pick)
            if not any(_is_variant(candidates[pick], candidates[o]) or _is_part_of(candidates[pick], candidates[o])
                       for o in chosen):
                chosen.append(pick)
        return [candidates[i] for i in chosen]
