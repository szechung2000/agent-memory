"""Provider token counters are read from responses, never guessed."""

from types import SimpleNamespace

import openai

from agent_memory.core.answer import OpenAIAnswerGenerator


def test_model_token_usage_tracks_provider_metadata(monkeypatch):
    payload = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":[]}'))],
        usage=SimpleNamespace(prompt_tokens=71, completion_tokens=9, total_tokens=80),
    )

    def fake_create(**kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        return payload

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: fake_client)
    generator = OpenAIAnswerGenerator(api_key="test-local-only")
    assert generator.generate("test", []) == '{"claims":[]}'
    assert generator.last_usage == {
        "prompt_tokens": 71,
        "completion_tokens": 9,
        "total_tokens": 80,
    }
