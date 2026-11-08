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

import pytest


def test_the_removed_llm_backend_fails_at_import_time(reload_iri_api):
    with pytest.raises(RuntimeError, match="'keyllm' was removed"):
        reload_iri_api(EXTRACTION_BACKEND="keyllm")


def test_an_unknown_backend_fails_at_import_time(reload_iri_api):
    with pytest.raises(RuntimeError, match="Invalid EXTRACTION_BACKEND 'bogus'"):
        reload_iri_api(EXTRACTION_BACKEND="bogus")


def test_keybert_is_accepted_explicitly(reload_iri_api):
    mod = reload_iri_api(EXTRACTION_BACKEND="keybert")

    assert mod.kw_model is not None


def test_the_default_bert_model_is_loaded_at_import_time(reload_iri_api, monkeypatch):
    monkeypatch.delenv("EXTRACTION_BACKEND", raising=False)
    mod = reload_iri_api()

    assert mod.kw_model is not None
    assert "pubmedbert" in mod._bert_model_cache
