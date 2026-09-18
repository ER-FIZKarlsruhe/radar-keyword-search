import httpx

from iri_api import search_tib_best_match


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
    }


async def test_returns_none_when_distance_exceeds_threshold():
    docs = [{"iri": "https://example.org/x", "label": "completely different term"}]
    async with _client(docs) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=2, client=client)

    assert result is None


async def test_skips_docs_whose_iri_is_unreachable():
    docs = [{"iri": "https://example.org/insulin", "label": "insulin"}]
    async with _client(docs, head_status=404) as client:
        result = await search_tib_best_match("insulin", None, None, threshold=5, client=client)

    assert result is None


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
