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

from iri_api import edit_distance, words_match


def test_identical_strings_have_zero_distance():
    assert edit_distance("insulin", "insulin") == 0


def test_is_case_insensitive():
    assert edit_distance("Insulin", "insulin") == 0


def test_counts_character_differences():
    assert edit_distance("insulin", "insular") == 2


def test_counts_inserted_characters_once():
    assert edit_distance("cat", "catch") == 2


def test_an_inserted_character_does_not_shift_the_rest():
    # Hamming distance compared position by position, so the one extra "e" made every later
    # character differ and "digital edition" could never match "Digitale Edition".
    assert edit_distance("digital edition", "Digitale Edition") == 1


def test_empty_strings_have_zero_distance():
    assert edit_distance("", "") == 0


def test_words_match_needs_the_same_number_of_words():
    assert not words_match("periodic dft", "Periodic")
    assert not words_match("witches", "Witches game")


def test_words_match_allows_a_quarter_of_each_words_length():
    assert words_match("digital edition", "Digitale Edition")
    assert words_match("cartesian coordinates", "Cartesian coordinate")
    assert not words_match("computational details", "Computational physics")
    # One word may not use up the whole phrase's allowance: "ir" (2 letters) must be exact.
    assert not words_match("ir spectroscopy", "uv spectroscopy")


def test_words_match_splits_hyphenated_words():
    # Otherwise the hyphenated word counts as one word, and dropping its first part ("o2-")
    # fits into the quarter allowed for the whole.
    assert not words_match("o2-reactivity", "Reactivity")
    assert words_match("anti-inflammatory drug", "anti-inflammatory drugs")
    assert words_match("anti-inflammatory", "anti inflammatory")
