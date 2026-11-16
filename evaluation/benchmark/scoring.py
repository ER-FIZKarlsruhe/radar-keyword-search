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

"""Scoring of keywords against reference keywords, shared by run.py and report.py.

A keyword matches a reference keyword if both have the same number of words and each word
pair is within the service's allowed edit distance (a quarter of the word's length), after
lowercasing and dropping GND qualifiers such as "Wien <Stadt>" or "Bank (Möbel)".
"""
import re

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


def normalize(term: str) -> list:
    term = re.sub(r"<[^>]*>|\([^)]*\)", " ", term.lower())
    return re.findall(r"\w+", term)


def matches(keyword: list, gold: list) -> bool:
    return len(keyword) == len(gold) and all(
        _edit_distance(k, g) <= len(g) // 4 for k, g in zip(keyword, gold)
    )


def score(keywords: list, gold: list) -> dict:
    kw = [normalize(k) for k in keywords]
    gd = [normalize(g) for g in gold if normalize(g)]
    hit_kw = sum(any(matches(k, g) for g in gd) for k in kw)
    hit_gold = sum(any(matches(k, g) for k in kw) for g in gd)
    p = hit_kw / len(kw) if kw else 0.0
    r = hit_gold / len(gd) if gd else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def gold_in_text(text: str, gold: list) -> float:
    """Share of the gold keywords that occur in the text (inflection-tolerant): the recall an
    extractive method could reach at best."""
    words = normalize(text)
    found = 0
    for g in gold:
        gw = normalize(g)
        if gw and any(matches(words[i:i + len(gw)], gw) for i in range(len(words) - len(gw) + 1)):
            found += 1
    return found / len(gold) if gold else 0.0
