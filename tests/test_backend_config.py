from unittest.mock import MagicMock

import pytest


def _fake_chat_completion(content: str):
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


def test_invalid_backend_raises_at_import_time(load_backend):
    with pytest.raises(RuntimeError, match="Invalid EXTRACTION_BACKEND"):
        load_backend("bogus")


def test_keybert_backend_does_not_talk_to_ollama(load_backend):
    mod = load_backend("keybert")

    assert mod.kw_model is not None
    assert mod.llm_client is None


def test_keyllm_backend_uses_local_endpoint_by_default(load_backend):
    mod = load_backend("keyllm")

    assert mod.kw_model is None
    assert mod.llm_client is not None
    assert mod.llm_model == "llama3"

    _, client_kwargs = mod.openai.OpenAI.call_args
    assert client_kwargs["base_url"] == "http://localhost:11434/v1"


def test_keyllm_backend_requests_structured_json_output(load_backend):
    # A chat model can ignore an informal "respond with ONLY the keywords,
    # separated by commas" instruction and answer conversationally instead
    # (e.g. "Here are the extracted keywords: cell"). response_format=
    # {"type": "json_object"} constrains the model's output via
    # grammar-constrained decoding, so there's no free-form prose for a
    # lead-in/sign-off sentence to leak through - see
    # _extract_keywords_via_ollama.
    mod = load_backend("keyllm")
    mod.llm_client.chat.completions.create.return_value = _fake_chat_completion('{"keywords": []}')

    mod._extract_keywords_via_ollama("some document")

    _, kwargs = mod.llm_client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "json" in mod.OLLAMA_KEYWORD_EXTRACTION_SYSTEM_PROMPT.lower()
    assert "keywords" in mod.OLLAMA_KEYWORD_EXTRACTION_SYSTEM_PROMPT.lower()


def test_keyllm_base_url_and_model_are_overridable(load_backend):
    mod = load_backend(
        "keyllm",
        OLLAMA_BASE_URL="http://gpu-box:11434/v1",
        OLLAMA_MODEL="mistral",
    )

    _, client_kwargs = mod.openai.OpenAI.call_args
    assert client_kwargs["base_url"] == "http://gpu-box:11434/v1"
    assert mod.llm_model == "mistral"


def test_keyllm_backend_bypasses_the_system_proxy_by_default(load_backend, monkeypatch):
    monkeypatch.delenv("OLLAMA_HTTP_PROXY", raising=False)
    captured_kwargs = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("httpx.Client", FakeHttpxClient)

    load_backend("keyllm")

    assert captured_kwargs == {"trust_env": False}


def test_keyllm_backend_uses_an_explicit_proxy_when_configured(load_backend, monkeypatch):
    captured_kwargs = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("httpx.Client", FakeHttpxClient)

    load_backend("keyllm", OLLAMA_HTTP_PROXY="http://proxy.example.com:8080")

    assert captured_kwargs == {"proxy": "http://proxy.example.com:8080"}
