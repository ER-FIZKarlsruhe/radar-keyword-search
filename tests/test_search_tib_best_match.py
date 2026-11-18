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
from iri_api import resolve_collection_ids, search_tib_best_match


def _client(docs, head_status=200):
    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(head_status)
        return httpx.Response(200, json={"response": {"docs": docs}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_returns_best_match_within_threshold():
    docs = [
        {
            "iri": "https://example.org/insulin",
            "label": "insulin",
            "synonym": ["Insulin hormone"],
            "ontology_name": "chebi",
        }
    ]
    async with _client(docs) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result == {
        "iri": "https://example.org/insulin",
        "label": "insulin",
        "best_term": "insulin",
        "distance": 0,
        "ontology_name": "chebi",
        "source": "tib",
        "keyword_scheme": "Other",
        "scheme_uri": "https://example.org/",
        "classification_code": "insulin",
        "matched": True,
    }


async def test_prefers_tibs_short_form_as_classification_code():
    docs = [{"iri": "http://purl.obolibrary.org/obo/CHEBI_5931", "label": "insulin", "short_form": "CHEBI_5931"}]
    async with _client(docs) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result["scheme_uri"] == "http://purl.obolibrary.org/obo/"
    assert result["classification_code"] == "CHEBI_5931"


async def test_marks_unmatched_when_distance_exceeds_threshold():
    docs = [{"iri": "https://example.org/x", "label": "completely different term"}]
    async with _client(docs) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=2, client=client)

    assert result["matched"] is False
    assert result["iri"] == "https://example.org/x"


async def test_marks_unmatched_when_the_best_candidates_iri_is_unreachable():
    docs = [{"iri": "https://example.org/insulin", "label": "insulin"}]
    async with _client(docs, head_status=404) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result["matched"] is False
    assert result["iri"] == "https://example.org/insulin"


async def test_returns_none_when_there_are_no_docs():
    async with _client([]) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result is None


async def test_returns_none_when_the_search_request_fails():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result is None


async def test_returns_none_when_the_search_request_returns_a_server_error():
    # A single term's TIB lookup returning a non-2xx status (e.g. a transient 500)
    # must not fail the whole /extract-iris batch - see asyncio.gather in extract_iris.
    def handler(request):
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result is None


async def test_includes_ontology_and_collection_filters_in_the_request():
    seen_urls = []

    def handler(request):
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"response": {"docs": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await search_tib_best_match("insulin", "chebi", "cs", threshold=5, client=client)

    assert "ontology=chebi" in seen_urls[0]
    assert "schema=collection" in seen_urls[0]
    assert "classification=cs" in seen_urls[0]


async def test_includes_collection_filter_without_an_ontology():
    # This is the shape radar-frontend actually sends: a workspace/contract can be
    # configured with only an ontology collection (no single ontology).
    seen_urls = []

    def handler(request):
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"response": {"docs": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await search_tib_best_match("apoptosis", None, "DataPLANT", threshold=5, client=client)

    assert "ontology=" not in seen_urls[0]
    assert "schema=collection" in seen_urls[0]
    # Passed through verbatim: TIB's filter is case-sensitive, "DATAPLANT" matches nothing
    assert "classification=DataPLANT" in seen_urls[0]


async def test_url_encodes_collection_ids_with_spaces():
    seen_urls = []

    def handler(request):
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"response": {"docs": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await search_tib_best_match("apoptosis", None, "FAIR Data Spaces,NFDI4CHEM", threshold=5, client=client)

    assert "classification=FAIR%20Data%20Spaces,NFDI4CHEM" in seen_urls[0]


def _by_collection_client(in_collection, everywhere, seen_urls):
    """TIB answering with `in_collection` docs for a collection-filtered search, else `everywhere`."""
    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(200)
        seen_urls.append(str(request.url))
        docs = in_collection if "classification=" in str(request.url) else everywhere
        return httpx.Response(200, json={"response": {"docs": docs}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


RDM = {"iri": "https://example.org/rdm", "label": "research data management", "ontology_name": "t4fs"}
DATA_MANAGEMENT = {"iri": "https://example.org/dm", "label": "Data management", "ontology_name": "edam"}


async def test_searches_all_of_tib_when_the_collection_has_no_match():
    seen_urls = []
    async with _by_collection_client([DATA_MANAGEMENT], [RDM], seen_urls) as client:
        result = await search_tib_best_match("research data management", None, "NFDI4CHEM", threshold=5, client=client)

    assert len(seen_urls) == 2
    assert "classification=" not in seen_urls[1]
    assert result["iri"] == "https://example.org/rdm"
    assert result["ontology_name"] == "t4fs"
    assert result["matched"] is True
    assert result["outside_collection"] is True


async def test_keeps_the_collection_match_without_a_second_search():
    seen_urls = []
    async with _by_collection_client([RDM], [], seen_urls) as client:
        result = await search_tib_best_match("research data management", None, "NFDI4CHEM", threshold=5, client=client)

    assert len(seen_urls) == 1
    assert result["matched"] is True
    assert "outside_collection" not in result


async def test_single_words_are_never_searched_outside_the_collection():
    # A single word matches some term in one of TIB's hundreds of ontologies far too easily
    # ("calculated", "reducing"); phrases are specific enough.
    research = {"iri": "https://example.org/research", "label": "Research", "ontology_name": "eclas"}
    seen_urls = []
    async with _by_collection_client([], [research], seen_urls) as client:
        result = await search_tib_best_match("research", None, "NFDI4CHEM", threshold=5, client=client)

    assert len(seen_urls) == 1
    assert result is None


async def test_prefers_the_collection_candidate_when_neither_matches():
    # Equally far in both searches: the collection's candidate is reported.
    in_collection = {"iri": "https://example.org/in", "label": "research data mgmt"}
    elsewhere = {"iri": "https://example.org/out", "label": "research data mgmt"}
    seen_urls = []
    async with _by_collection_client([in_collection], [elsewhere], seen_urls) as client:
        result = await search_tib_best_match("research data management", None, "NFDI4CHEM", threshold=5, client=client)

    assert len(seen_urls) == 2
    assert result["matched"] is False
    assert result["iri"] == "https://example.org/in"


async def test_an_explicit_ontology_is_never_left():
    seen_urls = []
    async with _by_collection_client([], [RDM], seen_urls) as client:
        result = await search_tib_best_match("research data management", "edam", "NFDI4CHEM", threshold=5, client=client)

    assert len(seen_urls) == 1
    assert result is None


async def test_no_second_search_without_a_collection():
    seen_urls = []
    async with _by_collection_client([], [DATA_MANAGEMENT], seen_urls) as client:
        await search_tib_best_match("research data management", None, None, threshold=5, client=client)

    assert len(seen_urls) == 1


def _collections_client(ids, status=200):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(status, json={"content": ids})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


async def test_resolve_collection_ids_restores_tib_casing(monkeypatch):
    monkeypatch.setattr(iri_api, "_tib_collection_ids", None)
    client, _ = _collections_client(["DataPLANT", "NFDI4Energy", "NFDI4CHEM", "FAIR Data Spaces"])
    async with client:
        result = await resolve_collection_ids("dataplant, NFDI4ENERGY,nfdi4chem,fair data spaces", client)

    assert result == "DataPLANT,NFDI4Energy,NFDI4CHEM,FAIR Data Spaces"


async def test_resolve_collection_ids_uppercases_unknown_ids_and_caches_the_list(monkeypatch):
    monkeypatch.setattr(iri_api, "_tib_collection_ids", None)
    client, calls = _collections_client(["NFDI4CHEM"])
    async with client:
        assert await resolve_collection_ids("nfdi4ing", client) == "NFDI4ING"
        assert await resolve_collection_ids("nfdi4chem", client) == "NFDI4CHEM"

    assert len(calls) == 1


async def test_resolve_collection_ids_falls_back_and_retries_when_the_list_fails(monkeypatch):
    monkeypatch.setattr(iri_api, "_tib_collection_ids", None)
    client, calls = _collections_client([], status=500)
    async with client:
        assert await resolve_collection_ids("nfdi4chem", client) == "NFDI4CHEM"
        assert await resolve_collection_ids("nfdi4chem", client) == "NFDI4CHEM"

    assert len(calls) == 2
    assert iri_api._tib_collection_ids is None


async def test_resolve_collection_ids_passes_through_empty_values():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        assert await resolve_collection_ids(None, client) is None
        assert await resolve_collection_ids("", client) == ""
        assert await resolve_collection_ids(" , ", client) is None
