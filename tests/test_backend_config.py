import pytest


def test_invalid_backend_raises_at_import_time(load_backend):
    with pytest.raises(RuntimeError, match="Invalid EXTRACTION_BACKEND"):
        load_backend("bogus")


def test_pubmedbert_backend_does_not_talk_to_ollama(load_backend):
    mod = load_backend("pubmedbert")

    assert mod.kw_model is not None
    assert mod.llm_kw_model is None


def test_ollama_backend_uses_local_endpoint_by_default(load_backend):
    mod = load_backend("ollama")

    assert mod.kw_model is None
    assert mod.llm_kw_model is not None

    _, client_kwargs = mod.openai.OpenAI.call_args
    assert client_kwargs["base_url"] == "http://localhost:11434/v1"
    _, wrapper_kwargs = mod.OpenAIWrapper.call_args
    assert wrapper_kwargs["model"] == "llama3"
    assert wrapper_kwargs["chat"] is True


def test_ollama_base_url_and_model_are_overridable(load_backend):
    mod = load_backend(
        "ollama",
        OLLAMA_BASE_URL="http://gpu-box:11434/v1",
        OLLAMA_MODEL="mistral",
    )

    _, client_kwargs = mod.openai.OpenAI.call_args
    assert client_kwargs["base_url"] == "http://gpu-box:11434/v1"
    _, wrapper_kwargs = mod.OpenAIWrapper.call_args
    assert wrapper_kwargs["model"] == "mistral"


def test_ollama_backend_bypasses_the_system_proxy_by_default(load_backend, monkeypatch):
    monkeypatch.delenv("OLLAMA_HTTP_PROXY", raising=False)
    captured_kwargs = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("httpx.Client", FakeHttpxClient)

    load_backend("ollama")

    assert captured_kwargs == {"trust_env": False}


def test_ollama_backend_uses_an_explicit_proxy_when_configured(load_backend, monkeypatch):
    captured_kwargs = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("httpx.Client", FakeHttpxClient)

    load_backend("ollama", OLLAMA_HTTP_PROXY="http://proxy.example.com:8080")

    assert captured_kwargs == {"proxy": "http://proxy.example.com:8080"}
