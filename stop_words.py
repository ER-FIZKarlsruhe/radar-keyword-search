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
# _SNOWBALL_GERMAN is the German stop word list of the Snowball project
# (https://snowballstem.org/algorithms/german/stop.txt), used under the
# 3-clause BSD License:
#
#   Copyright (c) 2001, Dr Martin Porter,
#   Copyright (c) 2002, Richard Boulton.
#   All rights reserved.
#
# The full license text is in the NOTICE file.
# ---------------------------------------------------------------------------
"""Stop words removed before keyword extraction: English and German.

RADAR metadata is mostly English or German, often both in one document (a
German abstract with an English translation). With English stop words only,
German filler words such as "hier", "und" or "der" became keywords.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SNOWBALL_GERMAN = frozenset({
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander",
    "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr",
    "anders", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "das",
    "dasselbe", "dazu", "daß", "dein", "deine", "deinem", "deinen", "deiner", "deines", "dem",
    "demselben", "den", "denn", "denselben", "der", "derer", "derselbe", "derselben", "des",
    "desselben", "dessen", "dich", "die", "dies", "diese", "dieselbe", "dieselben", "diesem",
    "diesen", "dieser", "dieses", "dir", "doch", "dort", "du", "durch", "ein", "eine", "einem",
    "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges",
    "einmal", "er", "es", "etwas", "euch", "euer", "eure", "eurem", "euren", "eurer", "eures",
    "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin",
    "hinter", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres",
    "im", "in", "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene",
    "jenem", "jenen", "jener", "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen",
    "keiner", "keines", "können", "könnte", "machen", "man", "manche", "manchem", "manchen",
    "mancher", "manches", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mich",
    "mir", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob",
    "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst",
    "sich", "sie", "sind", "so", "solche", "solchem", "solchen", "solcher", "solches", "soll",
    "sollte", "sondern", "sonst", "um", "und", "uns", "unse", "unsem", "unsen", "unser",
    "unses", "unter", "viel", "vom", "von", "vor", "war", "waren", "warst", "was", "weg",
    "weil", "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde",
    "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "wollte",
    "während", "würde", "würden", "zu", "zum", "zur", "zwar", "zwischen", "über"
})

# Our additions: the post-1996 spelling "dass" (Snowball only has "daß") and the
# ASCII spellings of the umlaut words, common in metadata ("fuer", "ueber").
_ADDED = frozenset({
    "dass", "fuer", "ueber", "koennen", "koennte", "waehrend", "wuerde", "wuerden",
})

# German stop words that are also meaningful English keywords, and stay
# candidates: "war" ("was" in German) is a central topic in history datasets.
_KEEP = frozenset({"war"})

GERMAN_STOP_WORDS = (_SNOWBALL_GERMAN | _ADDED) - _KEEP

# A sorted list, as scikit-learn's CountVectorizer expects.
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | GERMAN_STOP_WORDS)
