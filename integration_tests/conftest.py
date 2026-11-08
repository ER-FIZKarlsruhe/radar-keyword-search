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
# stubs (and its default `pytest` run) don't apply here; these tests talk to
# real external services and set up their own stubs (see real_tib_backend).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Shared by test_tib_service.py and test_gnd_service.py.
_STUBBED_MODULES = ("torch", "transformers", "keyword_extraction")


@pytest.fixture
def real_tib_backend(monkeypatch):
    # The BERT stack is stubbed, so no model is downloaded; the tests
    # monkeypatch _extract_keyword_list directly to choose the keywords, which
    # keeps them about the real TIB/GND lookups only.
    original_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    for name in _STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    monkeypatch.delenv("EXTRACTION_BACKEND", raising=False)

    import iri_api

    importlib.reload(iri_api)
    try:
        yield iri_api
    finally:
        # Restore whatever was really in sys.modules (or lack thereof) for
        # these names, so the stand-ins never leak beyond this fixture.
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
