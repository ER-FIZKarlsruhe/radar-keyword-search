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

"""
Shared test setup.

iri_api.py loads a real BERT model (via torch/transformers) *at import time*.
That is not desirable in a test run: we don't want tests to require model
downloads or network access.

So before iri_api is imported, we replace the heavy ML packages it depends on
(torch, transformers) with plain MagicMock stand-ins, along with
keyword_extraction (which needs numpy/scikit-learn; it has its own real
tests in test_keyword_extraction.py) and noun_phrases (which needs spaCy;
real tests in test_noun_phrases.py). Every attribute access / call on a
MagicMock just returns another MagicMock, so all of the module-level setup
code in iri_api.py (model loading etc.) runs without error. Individual tests
then configure the specific mock methods they need (e.g.
`iri_api.kw_model.extract_keywords.return_value = ...`).

Because configuration is read at *import* time, testing a different
configuration requires reloading the module (see the `reload_iri_api`
fixture) rather than just re-importing it.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest

for _module_name in ("torch", "transformers", "keyword_extraction", "noun_phrases"):
    sys.modules[_module_name] = MagicMock()

# scipy (used by the real scikit-learn in test_keyword_extraction.py) checks
# whether arrays are torch tensors whenever "torch" is already imported, via
# issubclass(..., torch.Tensor) - which needs a real class, not a MagicMock.
sys.modules["torch"].Tensor = type("Tensor", (), {})

import iri_api  # noqa: E402  (must be imported after the stubs above)


@pytest.fixture
def reload_iri_api(monkeypatch):
    """Reload iri_api with the given env vars set.

    Usage: `mod = reload_iri_api(BERT_MODEL="scibert")`
    """
    extra_keys = set()

    def _load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
            extra_keys.add(key)
        importlib.reload(iri_api)
        # MagicMock memoizes call_args/return_value/side_effect on the
        # stubbed keyword_extraction module, so the "new" kw_model
        # created by this reload can actually be the *same* mock object a
        # previous test configured. Start every reload from a clean slate.
        iri_api.kw_model.extract_keywords.reset_mock(return_value=True, side_effect=True)
        return iri_api

    yield _load

    # Clear any env vars a test passed in before restoring the default
    # configuration: monkeypatch only undoes them once this fixture (and
    # this reload) has finished, so a value that fails module-level
    # validation (e.g. an invalid BERT_MODEL) would otherwise still be set
    # here and break this restoring reload too, leaving iri_api broken for
    # every test that runs after this one.
    for key in extra_keys:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(iri_api)


@pytest.fixture(autouse=True)
def _reset_ml_mocks():
    """Ensure mock configuration from one test never leaks into the next."""
    yield
    iri_api.kw_model.extract_keywords.reset_mock(return_value=True, side_effect=True)
    iri_api.noun_phrase_candidates.reset_mock(return_value=True, side_effect=True)
