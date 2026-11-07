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

import httpx

import iri_api
from iri_api import pick_better_match, search_gnd_best_match

SCHAUSPIELMUSIK = {
    "id": "https://d-nb.info/gnd/4280160-6",
    "gndIdentifier": "4280160-6",
    "preferredName": "Schauspielmusik",
    "variantName": ["Theatermusik", "Bühnenmusik <Schauspielmusik>"],
    "type": ["AuthorityResource", "SubjectHeading"],
}


def _client(members, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={"totalItems": len(members), "member": members})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_returns_best_match_with_gnd_keyword_fields():
    async with _client([SCHAUSPIELMUSIK]) as client:
        result = await search_gnd_best_match("Schauspielmusik", threshold=5, client=client)

    assert result == {
        "iri": "https://d-nb.info/gnd/4280160-6",
        "label": "Schauspielmusik",
        "best_term": "Schauspielmusik",
        "distance": 0,
        "ontology_name": None,
        "source": "gnd",
        "keyword_scheme": "GND",
        "scheme_uri": "https://d-nb.info/gnd/",
        "classification_code": "4280160-6",
        "matched": True,
    }


async def test_matches_via_a_variant_name_but_reports_the_preferred_label():
    async with _client([SCHAUSPIELMUSIK]) as client:
        result = await search_gnd_best_match("theatermusik", threshold=5, client=client)

    assert result["best_term"] == "Theatermusik"
    assert result["label"] == "Schauspielmusik"
    assert result["distance"] == 0
    assert result["matched"] is True


async def test_marks_unmatched_when_distance_exceeds_threshold():
    async with _client([SCHAUSPIELMUSIK]) as client:
        result = await search_gnd_best_match("music", threshold=5, client=client)

    assert result["matched"] is False
    assert result["iri"] == "https://d-nb.info/gnd/4280160-6"


async def test_falls_back_to_the_iri_for_the_classification_code():
    member = {"id": "https://d-nb.info/gnd/4132033-5", "preferredName": "Edition"}
    async with _client([member]) as client:
        result = await search_gnd_best_match("Edition", threshold=5, client=client)

    assert result["classification_code"] == "4132033-5"


async def test_returns_none_when_there_are_no_members():
    async with _client([]) as client:
        assert await search_gnd_best_match("xyz", threshold=5, client=client) is None


async def test_returns_none_when_the_search_request_fails():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await search_gnd_best_match("Edition", threshold=5, client=client) is None


async def test_returns_none_on_a_non_json_reply():
    # e.g. lobid's bot-protection challenge page
    def handler(request):
        return httpx.Response(200, text="<html>Access Denied</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await search_gnd_best_match("Edition", threshold=5, client=client) is None


async def test_filters_by_the_configured_gnd_types(monkeypatch):
    monkeypatch.setattr(iri_api, "GND_TYPES", ["SubjectHeading", "Person"])
    seen = []
    async with _client([], seen) as client:
        await search_gnd_best_match("Goethe", threshold=5, client=client)

    params = seen[0].url.params
    assert params["q"] == "Goethe"
    assert params["format"] == "json"
    assert params["filter"] == "type:SubjectHeading OR type:Person"


async def test_sends_no_filter_when_no_types_are_configured(monkeypatch):
    monkeypatch.setattr(iri_api, "GND_TYPES", [])
    seen = []
    async with _client([], seen) as client:
        await search_gnd_best_match("Goethe", threshold=5, client=client)

    assert "filter" not in seen[0].url.params


def test_pick_better_match_prefers_matched_then_distance_then_first():
    tib = {"source": "tib", "matched": False, "distance": 1}
    gnd = {"source": "gnd", "matched": True, "distance": 3}
    assert pick_better_match(tib, gnd) is gnd

    tib = {"source": "tib", "matched": True, "distance": 2}
    gnd = {"source": "gnd", "matched": True, "distance": 0}
    assert pick_better_match(tib, gnd) is gnd

    gnd = {"source": "gnd", "matched": True, "distance": 2}
    assert pick_better_match(tib, gnd) is tib

    assert pick_better_match(None, gnd) is gnd
    assert pick_better_match(tib, None) is tib
    assert pick_better_match(None, None) is None
