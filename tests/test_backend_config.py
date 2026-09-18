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
