import httpx

from iri_api import check_iri_exists


async def test_returns_true_for_200_response():
    async def handler(request):
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("https://example.org/thing", client) is True


async def test_returns_false_for_non_200_response():
    async def handler(request):
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("https://example.org/missing", client) is False


async def test_returns_false_on_request_error():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("https://example.org/unreachable", client) is False
