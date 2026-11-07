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
Real end-to-end test of the GND lookup via lobid-gnd (https://lobid.org/gnd).

Same setup as test_tib_service.py (ML packages stubbed, keyword extraction
faked); needs outbound network access to lobid.org. The expected entries are
taken from a real RADAR4Culture dataset's GND keywords
(https://radar4culture.radar-service.eu/radar/de/dataset/uupxmgkq94r0jpsk).
"""
from fastapi.testclient import TestClient



def test_extract_iris_resolves_real_gnd_subject_headings(real_tib_backend, monkeypatch):
    monkeypatch.setattr(
        real_tib_backend,
        "_extract_keyword_list",
        lambda document, bert_model=None, max_keywords=None: ["Schauspielmusik", "Forschungsdatenmanagement"],
    )

    with TestClient(real_tib_backend.app) as client:
        response = client.post("/extract-iris", json={"document": "...", "terminology": "gnd"})

    assert response.status_code == 200
    body = response.json()

    assert body["Schauspielmusik"]["iri"] == "https://d-nb.info/gnd/4280160-6"
    assert body["Forschungsdatenmanagement"]["iri"] == "https://d-nb.info/gnd/1342271416"
    for match in body.values():
        assert match["matched"] is True
        assert match["distance"] == 0
        assert match["source"] == "gnd"
        assert match["keyword_scheme"] == "GND"
        assert match["scheme_uri"] == "https://d-nb.info/gnd/"
        assert match["iri"] == match["scheme_uri"] + match["classification_code"]


def test_extract_iris_with_both_picks_per_keyword(real_tib_backend, monkeypatch):
    monkeypatch.setattr(
        real_tib_backend,
        "_extract_keyword_list",
        lambda document, bert_model=None, max_keywords=None: ["Schauspielmusik", "insulin"],
    )

    with TestClient(real_tib_backend.app) as client:
        response = client.post(
            "/extract-iris",
            json={"document": "...", "terminology": "both", "ontology": "chebi"},
        )

    assert response.status_code == 200
    body = response.json()
    # Only GND has a German music subject heading ...
    assert body["Schauspielmusik"]["source"] == "gnd"
    # ... while both have an exact "insulin"; on a tie TIB (the workspace's ontologies) wins.
    assert body["insulin"]["source"] == "tib"
    assert body["insulin"]["ontology_name"] == "chebi"
