"""Provider adapter + Azure reasoning-field normalization."""

from pathlib import Path

from prognosis.common.provider import chat_completion_payload, deployment_model_id, get_provider
from prognosis.agent.parse_mini_swe_traj import _assistant_message, parse_mini_swe_traj


def test_get_provider_default_vllm(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    assert get_provider() == "vllm"


def test_deployment_model_id_strips_prefix():
    assert deployment_model_id("openai/qwen3-32b") == "qwen3-32b"
    assert deployment_model_id("hosted_vllm/Qwen3-8B") == "Qwen3-8B"
    assert deployment_model_id("qwen3-32b") == "qwen3-32b"


def test_azure_payload_no_forbidden_kwargs_and_no_think():
    payload = chat_completion_payload(
        model="openai/qwen3-32b",
        messages=[{"role": "user", "content": "Give P(success)"}],
        temperature=0.0,
        max_tokens=32,
        enable_thinking=False,
        provider="azure",
    )
    assert payload["model"] == "qwen3-32b"
    assert "chat_template_kwargs" not in payload
    assert "enable_thinking" not in payload
    assert payload["messages"][-1]["content"].endswith("/no_think")


def test_vllm_payload_sends_chat_template_kwargs():
    payload = chat_completion_payload(
        model="hosted_vllm/Qwen3-8B",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=32,
        enable_thinking=True,
        provider="vllm",
    )
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_openrouter_payload_preserves_org_slash_and_uses_no_think():
    payload = chat_completion_payload(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": "Give P(success)"}],
        temperature=0.0,
        max_tokens=32,
        enable_thinking=False,
        provider="openrouter",
    )
    # Unlike Azure's litellm-style "openai/qwen3-32b", the org/model slash in
    # an OpenRouter id IS the model id -- must not be stripped.
    assert payload["model"] == "qwen/qwen3-32b"
    assert "chat_template_kwargs" not in payload
    assert payload["messages"][-1]["content"].endswith("/no_think")


def test_openrouter_payload_respects_enable_thinking_true():
    payload = chat_completion_payload(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=32,
        enable_thinking=True,
        provider="openrouter",
    )
    assert not payload["messages"][-1]["content"].endswith("/no_think")


def test_get_provider_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        get_provider("bogus")


def test_assistant_message_maps_azure_reasoning():
    msg = _assistant_message(
        {
            "role": "assistant",
            "content": "pong",
            "reasoning": "thinking about pong",
        }
    )
    assert msg["reasoning_content"] == "thinking about pong"
    assert "reasoning" not in msg


def test_parse_fixture_still_works():
    path = Path(__file__).resolve().parents[1] / "fixtures" / "sample_mini.traj.json"
    if not path.exists():
        return
    rec = parse_mini_swe_traj(path, outcome=1)
    assert rec.n_steps >= 1
