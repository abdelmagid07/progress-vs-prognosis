"""Resolve mini-swe-agent run config for a model provider (vllm | azure).

Used by ``scripts/run_mini_swe_batch.sh``. Keeps provider-specific request
shapes out of the shell script so switching backends is one env var.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import minisweagent
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mini-swe-agent not installed. pip install -e \".[agent]\""
    ) from exc


def deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve(
    *,
    override_cfg: Path,
    provider_cfg: Path,
    resolved_config: Path,
    model_name: str,
    api_base: str,
    api_key: str,
    step_limit: str | None,
    seed: str | None,
) -> dict[str, Any]:
    base_path = (
        Path(minisweagent.__file__).parent / "config" / "benchmarks" / "swebench.yaml"
    )
    cfg = deep_merge(load_yaml(base_path), load_yaml(override_cfg))
    provider = load_yaml(provider_cfg)
    # Provider file may carry metadata keys; only merge the model: block into cfg.
    if "model" in provider:
        deep_merge(cfg.setdefault("model", {}), provider["model"])

    model = cfg.setdefault("model", {})
    model_kwargs = model.setdefault("model_kwargs", {})
    model["model_name"] = model_name
    model_kwargs["api_base"] = api_base
    model_kwargs["api_key"] = api_key

    if step_limit and step_limit.strip():
        cfg.setdefault("agent", {})["step_limit"] = int(step_limit)

    if seed and seed.strip():
        model_kwargs.setdefault("extra_body", {})["seed"] = int(seed)

    # Strip any accidental vLLM-only keys when provider is azure.
    provider_name = str(provider.get("provider", "")).lower()
    if provider_name == "azure":
        extra = model_kwargs.get("extra_body") or {}
        extra.pop("top_k", None)
        extra.pop("chat_template_kwargs", None)
        extra.pop("enable_thinking", None)
        model_kwargs["extra_body"] = extra
        # Also strip if someone put them at model_kwargs top level.
        model_kwargs.pop("chat_template_kwargs", None)
        model_kwargs.pop("enable_thinking", None)
        model_kwargs.pop("top_k", None)

    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    resolved_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    thinking_mode = str(provider.get("thinking_mode", "explicit"))
    explicit_on = (
        (model_kwargs.get("extra_body") or {})
        .get("chat_template_kwargs", {})
        .get("enable_thinking")
        is True
    )

    print(f"Resolved config -> {resolved_config}")
    print(f"  base:            {base_path}")
    print(f"  provider:        {provider_name or provider_cfg.stem} ({provider_cfg})")
    print(f"  model_name:      {model['model_name']}")
    print(f"  api_base:        {model_kwargs['api_base']}")
    print(f"  seed:            {seed or '(unset)'}")
    print(f"  thinking_mode:   {thinking_mode}")
    print(f"  enable_thinking: {explicit_on}")

    if thinking_mode == "explicit" and not explicit_on:
        raise SystemExit(
            "Refusing to run: provider requires chat_template_kwargs.enable_thinking=true "
            f"(provider={provider_name!r})."
        )
    if thinking_mode == "default_on":
        print(
            "  note: Azure/Foundry thinking is server-default ON; "
            "do not send chat_template_kwargs / enable_thinking / top_k."
        )

    return cfg


def main() -> None:
    resolve(
        override_cfg=Path(os.environ["OVERRIDE_CFG"]),
        provider_cfg=Path(os.environ["PROVIDER_CFG"]),
        resolved_config=Path(os.environ["RESOLVED_CONFIG"]),
        model_name=os.environ["MODEL_NAME"],
        api_base=os.environ["MODEL_API_BASE"],
        api_key=os.environ["MODEL_API_KEY"],
        step_limit=os.environ.get("STEP_LIMIT"),
        seed=os.environ.get("SEED"),
    )


if __name__ == "__main__":
    main()
