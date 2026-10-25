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
Unit tests for _extract_keywords_via_ollama's JSON handling, isolated from the
/extract-iris endpoint and the TIB lookup. See tests/test_endpoints.py for the
end-to-end request/response shape of the keyllm backend.
"""
from unittest.mock import MagicMock


def _fake_chat_completion(content: str):
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


def test_returns_the_keywords_array_from_a_well_formed_reply(load_backend):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion(
        '{"keywords": ["insulin", "glucose"]}'
    )

    assert mod._extract_keywords_via_ollama("some document") == ["insulin", "glucose"]


def test_strips_and_drops_blank_entries(load_backend):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion(
        '{"keywords": ["  insulin  ", "", "   ", "glucose"]}'
    )

    assert mod._extract_keywords_via_ollama("some document") == ["insulin", "glucose"]


def test_returns_an_empty_list_when_keywords_is_missing(load_backend):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion("{}")

    assert mod._extract_keywords_via_ollama("some document") == []


def test_returns_an_empty_list_when_the_top_level_json_is_not_an_object(load_backend):
    # A model constrained to json_object should never actually emit a bare
    # array, but treat any shape other than {"keywords": [...]} as no
    # keywords found rather than crashing.
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('["insulin", "glucose"]')

    assert mod._extract_keywords_via_ollama("some document") == []


def test_raises_when_the_reply_is_not_valid_json(load_backend):
    import json

    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion("not json")

    try:
        mod._extract_keywords_via_ollama("some document")
        assert False, "expected a JSONDecodeError"
    except json.JSONDecodeError:
        pass


def test_sends_the_document_and_requests_json_output(load_backend):
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": []}')

    mod._extract_keywords_via_ollama("insulin regulates blood glucose")

    _, kwargs = mod.llm_client.chat.completions.create.call_args
    assert kwargs["model"] == mod.llm_model
    assert kwargs["response_format"] == {"type": "json_object"}
    user_message = kwargs["messages"][-1]["content"]
    assert "insulin regulates blood glucose" in user_message
