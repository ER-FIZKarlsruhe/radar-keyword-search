from iri_api import hamming_distance


def test_identical_strings_have_zero_distance():
    assert hamming_distance("insulin", "insulin") == 0


def test_is_case_insensitive():
    assert hamming_distance("Insulin", "insulin") == 0


def test_counts_character_differences():
    assert hamming_distance("insulin", "insular") == 2


def test_pads_the_shorter_string_with_spaces():
    # "cat" is right-padded to "cat  " (5 chars) to compare against "catch",
    # so the two trailing space-vs-letter positions count as differences.
    assert hamming_distance("cat", "catch") == 2


def test_empty_strings_have_zero_distance():
    assert hamming_distance("", "") == 0
