"""
Real end-to-end test of the TIB Terminology Service integration.

Unlike tests/test_endpoints.py (which fakes search_tib_best_match so the fast
suite never needs network access), this sends actual requests to the real
https://api.terminology.tib.eu API and does a real HEAD request against the
IRI it returns. It needs network access to both of those, so - like the
Ollama backend test in this directory - it's kept out of the default `pytest`
run (see pytest.ini's `testpaths = tests`).

Keyword extraction itself is faked, and the ML packages it would normally
need (torch/transformers/keybert/openai) are stubbed out exactly like
tests/conftest.py does for the fast suite - this test is about the TIB
lookup pipeline (which only depends on httpx), not about any particular
extraction backend, so it doesn't need the real ML stack installed at all.
The stubs are swapped back out afterwards so they don't leak into
test_ollama_backend.py, which needs the real packages.
"""
import importlib
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

_STUBBED_MODULES = ("torch", "transformers", "keybert", "keybert.llm", "openai")


@pytest.fixture
def real_tib_backend(monkeypatch):
    # The "ollama" backend is used purely as a lightweight way to get a
    # module state whose extract_keywords is easy to fake (see below) -
    # llm_kw_model's OpenAI client is constructed but never actually called,
    # so no Ollama server needs to be running for this.
    original_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    for name in _STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    monkeypatch.setenv("EXTRACTION_BACKEND", "ollama")

    import iri_api

    importlib.reload(iri_api)
    try:
        yield iri_api
    finally:
        # Restore whatever was really in sys.modules (or lack thereof) for
        # these names, so a later test in the same session - e.g.
        # test_ollama_backend.py's real Ollama/keybert/openai usage - never
        # sees our stand-ins. Deliberately doesn't reload iri_api back to
        # pubmedbert here: in the real integration environment that would
        # trigger a genuine PubMedBERT download, and no other test in this
        # package depends on iri_api's module-level state afterwards.
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_extract_iris_resolves_real_chebi_matches_via_the_tib_service(real_tib_backend):
    real_tib_backend.llm_kw_model.extract_keywords.return_value = ["insulin", "glucose"]

    with TestClient(real_tib_backend.app) as client:
        response = client.post(
            "/extract-iris",
            json={"document": "insulin and glucose", "ontology": "chebi", "ontology_collection": "cs"},
        )

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"insulin", "glucose"}
    for keyword, match in body.items():
        assert match is not None, f"expected a real ChEBI match for {keyword!r} from the TIB service"
        assert match["ontology_name"] == "chebi"
        assert match["iri"].startswith("http://purl.obolibrary.org/obo/CHEBI_")
        assert match["distance"] == 0
        assert match["best_term"] == keyword
