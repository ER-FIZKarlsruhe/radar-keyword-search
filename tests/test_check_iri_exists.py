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
import pytest

from iri_api import check_iri_exists


async def test_returns_true_for_200_response():
    async def handler(request):
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("https://example.org/thing", client) is True


@pytest.mark.parametrize("status", [404, 410])
async def test_returns_false_for_a_missing_iri(status):
    async def handler(request):
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("https://example.org/missing", client) is False


@pytest.mark.parametrize("status", [401, 403, 405, 429, 500, 503])
async def test_returns_true_when_the_answer_does_not_prove_the_iri_dead(status):
    # e.g. www.loc.gov (PREMIS) answers automated clients with 403, and with a 503 challenge
    # after a burst of requests
    async def handler(request):
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("http://www.loc.gov/premis/rdf/v3/Fixity", client) is True


async def test_returns_true_on_a_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("http://www.loc.gov/premis/rdf/v3/Fixity", client) is True


async def test_returns_false_when_the_host_is_unreachable():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_iri_exists("https://example.org/unreachable", client) is False
