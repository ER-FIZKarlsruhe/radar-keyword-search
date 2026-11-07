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

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# This directory is intentionally NOT under tests/, so tests/conftest.py's
# stubbing of torch/transformers/keybert/openai never applies here - these
# tests need the real packages to talk to a real Ollama server.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Shared by test_tib_service.py and test_gnd_service.py.
_STUBBED_MODULES = ("torch", "transformers", "keybert", "openai")


@pytest.fixture
def real_tib_backend(monkeypatch):
    # The "keyllm" backend is used purely as a lightweight way to get a
    # module state whose keyword extraction is easy to fake (see below) - by
    # monkeypatching _extract_keyword_list directly, the OpenAI client is
    # constructed but never actually called, so no Ollama server needs to be
    # running for this.
    original_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    for name in _STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    monkeypatch.setenv("EXTRACTION_BACKEND", "keyllm")

    import iri_api

    importlib.reload(iri_api)
    try:
        yield iri_api
    finally:
        # Restore whatever was really in sys.modules (or lack thereof) for
        # these names, so a later test in the same session - e.g.
        # test_keyllm_backend.py's real Ollama/keybert/openai usage - never
        # sees our stand-ins. Deliberately doesn't reload iri_api back to
        # keybert here: in the real integration environment that would
        # trigger a genuine PubMedBERT download, and no other test in this
        # package depends on iri_api's module-level state afterwards.
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
