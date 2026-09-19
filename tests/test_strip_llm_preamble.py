from iri_api import _strip_llm_preamble


def test_strips_a_conversational_lead_in_sentence():
    assert _strip_llm_preamble("Here are the extracted keywords: cell") == "cell"


def test_strips_a_lead_in_even_with_extra_whitespace():
    assert _strip_llm_preamble("Sure, here are the keywords:   apoptosis  ") == "apoptosis"


def test_leaves_a_bare_keyword_untouched():
    assert _strip_llm_preamble("mitochondria") == "mitochondria"


def test_leaves_a_keyword_with_no_space_before_the_colon_untouched():
    # A compact identifier-like token (no spaces before the colon) is not a
    # sentence lead-in, so it must not be mistaken for one.
    assert _strip_llm_preamble("CHEBI:145810") == "CHEBI:145810"


def test_leaves_a_colon_with_nothing_useful_after_it_untouched():
    assert _strip_llm_preamble("Here are the extracted keywords:") == "Here are the extracted keywords:"
