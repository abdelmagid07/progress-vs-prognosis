"""Model-provider helpers (vLLM vs Azure Foundry vs OpenRouter).

Generation request shape lives in ``configs/providers/*.yaml`` and is applied by
``scripts/resolve_mini_config.py``. This module covers the smaller call sites
(elicitation, model-id normalization) so switching providers stays one env var:
``MODEL_PROVIDER=vllm|azure|openrouter``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

Provider = Literal["vllm", "azure", "openrouter"]


def get_provider(name: str | None = None) -> Provider:
    key = (name or os.environ.get("MODEL_PROVIDER") or "vllm").strip().lower()
    if key not in ("vllm", "azure", "openrouter"):
        raise ValueError(f"Unknown MODEL_PROVIDER={key!r}; use vllm, azure, or openrouter")
    return key  # type: ignore[return-value]


def deployment_model_id(model: str) -> str:
    """Strip litellm provider prefixes for raw OpenAI-compatible HTTP calls.

    ``openai/qwen3-32b`` / ``hosted_vllm/Qwen3-8B`` → deployment / served name.
    Not used for OpenRouter: there the ``org/model`` slash is part of the
    actual model id (e.g. ``qwen/qwen3-32b``), not a prefix to strip.
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def chat_completion_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    provider: Provider | None = None,
) -> dict[str, Any]:
    """Build a chat/completions JSON body accepted by the active provider.

    Azure Foundry rejects ``chat_template_kwargs`` / ``enable_thinking`` /
    ``top_k``. Thinking is server-default ON; for elicitation (thinking off) we
    append the Qwen soft-switch ``/no_think`` to the last user turn.

    OpenRouter uses the same ``/no_think`` soft-switch as Azure, not
    ``chat_template_kwargs`` — OpenRouter proxies to an upstream inference
    backend we don't control, so there's no guarantee it forwards a
    vLLM-specific request field. ``/no_think`` is a Qwen3 chat-template
    convention embedded in the text itself, so it works regardless of what's
    serving the request underneath.
    """
    prov = get_provider(provider)
    msgs = [dict(m) for m in messages]

    if prov in ("azure", "openrouter"):
        if not enable_thinking and msgs:
            # Soft-switch; validated empty reasoning with /no_think on Foundry.
            last = msgs[-1]
            if last.get("role") == "user":
                content = (last.get("content") or "").rstrip()
                if "/no_think" not in content and "/think" not in content:
                    last["content"] = f"{content}\n/no_think"
        return {
            "model": model if prov == "openrouter" else deployment_model_id(model),
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    # vLLM / self-hosted OpenAI-compatible
    return {
        "model": deployment_model_id(model),
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
