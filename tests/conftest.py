"""
Shared test setup.

iri_api.py loads a real PubMedBERT model from HuggingFace and a real OpenAI
client at *import time*, and refuses to import at all unless CHAT_GPT_API_KEY
is set. None of that is desirable in a test run: we don't want tests to
require a GPU, download a model, need network access, or need a real API key.

So before iri_api is imported, we replace the heavy ML packages it depends on
(torch, transformers, keybert, openai) with plain MagicMock stand-ins. Every
attribute access / call on a MagicMock just returns another MagicMock, so all
of the module-level setup code in iri_api.py (model loading, client
construction, etc.) runs without error. Individual tests then configure the
specific mock methods they need (e.g. `iri_api.kw_model.extract_keywords.
return_value = ...`).
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("CHAT_GPT_API_KEY", "test-api-key")

for _module_name in ("torch", "transformers", "keybert", "keybert.llm", "openai", "numpy"):
    sys.modules[_module_name] = MagicMock()

import iri_api  # noqa: E402  (must be imported after the stubs above)


@pytest.fixture(autouse=True)
def _reset_ml_mocks():
    """Ensure mock configuration from one test never leaks into the next."""
    yield
    iri_api.kw_model.extract_keywords.reset_mock(return_value=True, side_effect=True)
    iri_api.openai_kw_model.extract_keywords.reset_mock(return_value=True, side_effect=True)
