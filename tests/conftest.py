"""
Shared test setup.

iri_api.py picks its extraction backend (bert / ollama) from
the EXTRACTION_BACKEND env var *at import time*, and loads a real PubMedBERT
model or a real OpenAI client accordingly. None of that is desirable in a
test run: we don't want tests to require a GPU, download a model, need
network access, or need a running Ollama server.

So before iri_api is imported, we replace the heavy ML packages it depends on
(torch, transformers, keybert, openai) with plain MagicMock stand-ins. Every
attribute access / call on a MagicMock just returns another MagicMock, so all
of the module-level setup code in iri_api.py (model loading, client
construction, etc.) runs without error. Individual tests then configure the
specific mock methods they need (e.g. `iri_api.kw_model.extract_keywords.
return_value = ...`).

Because the backend is chosen at *import* time, switching backends between
tests requires reloading the module (see the `load_backend` fixture) rather
than just re-importing it.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest

for _module_name in ("torch", "transformers", "keybert", "openai", "numpy"):
    sys.modules[_module_name] = MagicMock()

os.environ.setdefault("EXTRACTION_BACKEND", "bert")

import iri_api  # noqa: E402  (must be imported after the stubs above)


@pytest.fixture
def load_backend(monkeypatch):
    """Reload iri_api with a specific EXTRACTION_BACKEND (and other env vars).

    Usage: `mod = load_backend("ollama", OLLAMA_MODEL="mistral")`
    """
    extra_keys = set()

    def _load(backend, **env):
        monkeypatch.setenv("EXTRACTION_BACKEND", backend)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
            extra_keys.add(key)
        importlib.reload(iri_api)
        # MagicMock memoizes call_args/return_value/side_effect on the
        # stubbed keybert/openai modules, so the "new" kw_model/llm_client
        # created by this reload can actually be the *same* mock object a
        # previous test configured. Start every reload from a clean slate.
        if iri_api.kw_model is not None:
            iri_api.kw_model.extract_keywords.reset_mock(return_value=True, side_effect=True)
        if iri_api.llm_client is not None:
            iri_api.llm_client.chat.completions.create.reset_mock(return_value=True, side_effect=True)
        return iri_api

    yield _load

    # Clear any extra env vars a test passed in before restoring the default
    # bert backend: monkeypatch only undoes them once this fixture (and this
    # reload) has finished, so a value that fails module-level validation
    # (e.g. an invalid BERT_MODEL) would otherwise still be set here and
    # break this restoring reload too, leaving iri_api broken for every test
    # that runs after this one.
    for key in extra_keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EXTRACTION_BACKEND", "bert")
    importlib.reload(iri_api)


@pytest.fixture(autouse=True)
def _reset_ml_mocks():
    """Ensure mock configuration from one test never leaks into the next."""
    yield
    if iri_api.kw_model is not None:
        iri_api.kw_model.extract_keywords.reset_mock(return_value=True, side_effect=True)
    if iri_api.llm_client is not None:
        iri_api.llm_client.chat.completions.create.reset_mock(return_value=True, side_effect=True)
