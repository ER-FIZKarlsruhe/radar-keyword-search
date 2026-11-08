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

"""Limits that keep one request from exhausting the service: the document size cap,
batched embedding, and keyword extraction off the event loop, one at a time."""
import asyncio
import threading
import time

import httpx
from fastapi.testclient import TestClient

import iri_api


# -------------------------------
# Document size
# -------------------------------
def test_rejects_a_document_longer_than_the_limit(reload_iri_api):
    mod = reload_iri_api(MAX_DOCUMENT_CHARS="20")

    with TestClient(mod.app) as client:
        response = client.post("/extract-iris", json={"document": "x" * 21})

    assert response.status_code == 422
    mod.kw_model.extract_keywords.assert_not_called()


def test_accepts_a_document_of_exactly_the_limit(reload_iri_api, monkeypatch):
    mod = reload_iri_api(MAX_DOCUMENT_CHARS="20")
    mod.kw_model.extract_keywords.return_value = []

    with TestClient(mod.app) as client:
        response = client.post("/extract-iris", json={"document": "x" * 20})

    assert response.status_code == 200


def test_the_default_limit_leaves_room_for_the_frontends_longest_documents():
    # radar-frontend sends title + descriptions + a README of up to 20,000 characters.
    assert iri_api.MAX_DOCUMENT_CHARS == 100_000


# -------------------------------
# Batched embedding
# -------------------------------
def test_embeds_candidates_in_batches():
    embedding = iri_api._BertEmbedding("some/model")
    iri_api.torch.cat.reset_mock()
    words = [f"word{i}" for i in range(2 * iri_api.EMBED_BATCH_SIZE + 5)]

    embedding.embed(words)

    batch_sizes = [len(c.args[0]) for c in embedding._tokenizer.call_args_list[-3:]]
    assert batch_sizes == [iri_api.EMBED_BATCH_SIZE, iri_api.EMBED_BATCH_SIZE, 5]
    assert all(len(c.args[0]) <= iri_api.EMBED_BATCH_SIZE for c in embedding._tokenizer.call_args_list)
    (batches,), _ = iri_api.torch.cat.call_args
    assert len(batches) == 3


# -------------------------------
# Extraction off the event loop
# -------------------------------
def _async_client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=iri_api.app), base_url="http://test")


async def _no_match(*args, **kwargs):
    return None


async def test_a_slow_extraction_does_not_block_other_requests(monkeypatch):
    def slow_extraction(*args, **kwargs):
        time.sleep(1.0)
        return [("insulin", 0.9)]

    iri_api.kw_model.extract_keywords.side_effect = slow_extraction
    monkeypatch.setattr(iri_api, "search_tib_best_match", _no_match)

    async with _async_client() as client:
        started = time.monotonic()
        extraction = asyncio.create_task(client.post("/extract-iris", json={"document": "insulin"}))
        await asyncio.sleep(0.1)  # let the extraction start
        health = await client.get("/")
        # Timed from the start of the extraction: if it blocked the event loop, neither the
        # sleep above nor the health check could finish before the 1 s extraction does.
        health_seconds = time.monotonic() - started

        assert health.status_code == 200
        assert health_seconds < 0.6
        assert not extraction.done()
        assert (await extraction).json() == {"insulin": None}


async def test_extractions_run_one_at_a_time(monkeypatch):
    active, peak = 0, 0
    guard = threading.Lock()

    def tracked_extraction(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.2)
        with guard:
            active -= 1
        return [("insulin", 0.9)]

    iri_api.kw_model.extract_keywords.side_effect = tracked_extraction
    monkeypatch.setattr(iri_api, "search_tib_best_match", _no_match)

    async with _async_client() as client:
        responses = await asyncio.gather(*(client.post("/extract-iris", json={"document": "insulin"}) for _ in range(3)))

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert peak == 1
