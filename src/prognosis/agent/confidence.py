#!/usr/bin/env python
"""Post-hoc verbalized P(success) elicitation over saved trajectory prefixes.

After trajectories are generated, take each saved prefix ending at step ``t``,
append an elicitation prompt in a *separate* forward pass, and parse a single
number in ``[0, 1]``. Eliciting post hoc rather than inline keeps the
trajectories under study uncontaminated.

Writes ``confidence.parquet`` keyed by ``trajectory_id``, ``step_index``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from prognosis.common.config import load_defaults
from prognosis.common.paths import NORMALIZED_DIR, config_file, data_file
from prognosis.common.provider import chat_completion_payload, get_provider
from prognosis.agent.schema import TrajectoryRecord, load_trajectories_from_dir

_PROB_RE = re.compile(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)")


def load_elicitation_prompt(path: Path | None = None) -> str:
    path = path or config_file("elicitation_prompt.txt")
    return Path(path).read_text(encoding="utf-8").strip()


def parse_probability(text: str) -> float | None:
    """Extract the first number in ``[0, 1]`` from a model reply; else ``None``."""
    if not text:
        return None
    for m in _PROB_RE.finditer(text.strip()):
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val
    return None


def _prefix_messages(record: TrajectoryRecord, step_index: int) -> list[dict]:
    """Messages the model saw through the end of step ``step_index`` (inclusive)."""
    step = record.steps[step_index]
    assistant_msg = step.assistant_message or {
        "role": "assistant",
        "content": step.assistant_response,
    }
    # Drop reasoning_content from history turns (Qwen3 template strips prior
    # think blocks); keep the current turn's native form for the elicitation
    # context so the model sees what it just produced.
    return list(step.messages_before_gen) + [assistant_msg]


def _chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    enable_thinking: bool = False,
    provider: str | None = None,
    timeout_s: int = 180,
) -> str:
    """One chat completion; returns content (thinking off by default for elicitation).

    Thinking is OFF for the elicitation pass: we want a short numeric answer, not
    a long ``<think>`` block. The *trajectory* under study was generated
    thinking-on; this is a separate post-hoc pass over the same prefix.

    Provider (``MODEL_PROVIDER`` / ``provider``) selects the request shape:
    vLLM gets ``chat_template_kwargs``; Azure gets a ``/no_think`` soft-switch
    (Foundry rejects the kwargs).
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    # Strip to OpenAI-compatible fields the endpoint accepts.
    clean: list[dict] = []
    for m in messages:
        out = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out["tool_calls"] = m["tool_calls"]
        if m.get("role") == "tool" and m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
        clean.append(out)
    payload = chat_completion_payload(
        model=model,
        messages=clean,
        temperature=temperature,
        max_tokens=32,
        enable_thinking=enable_thinking,
        provider=get_provider(provider),
    )
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    message = body["choices"][0]["message"]
    return (message.get("content") or "").strip()


def _elicit_one_step(
    record: TrajectoryRecord,
    step_index: int,
    *,
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    provider: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Elicit P(success) for a single (trajectory, step) — the unit of concurrency."""
    step = record.steps[step_index]
    n_steps = record.n_steps
    raw = ""
    p: float | None

    if dry_run:
        # Deterministic stand-in for offline wiring: rising toward outcome.
        p = 0.2 + 0.6 * (step_index / max(n_steps - 1, 1))
        if record.outcome == 0:
            p = 1.0 - p
        raw = f"{p:.2f}"
    else:
        messages = _prefix_messages(record, step_index) + [
            {"role": "user", "content": prompt},
        ]
        try:
            raw = _chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                provider=provider,
            )
        except Exception as exc:  # noqa: BLE001 - one bad call must not sink the batch
            print(
                f"  WARN {record.trajectory_id} step {step_index}: elicitation failed "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raw = ""
        p = parse_probability(raw)

    return {
        "task_id": record.task_id,
        "trajectory_id": record.trajectory_id,
        "seed": record.seed,
        "outcome": record.outcome,
        "step_index": step_index,
        "n_steps": n_steps,
        "rel_pos": 0.0 if n_steps <= 1 else step_index / (n_steps - 1),
        "p_success": p,
        "raw_response": raw,
    }


def elicit_for_trajectory(
    record: TrajectoryRecord,
    *,
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    provider: str | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Elicit P(success) at every step of one trajectory, sequentially."""
    return [
        _elicit_one_step(
            record,
            step.step_index,
            prompt=prompt,
            api_base=api_base,
            api_key=api_key,
            model=model,
            provider=provider,
            dry_run=dry_run,
        )
        for step in record.steps
    ]


def run(
    traj_dir: Path,
    *,
    output_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    prompt_path: Path | None = None,
    provider: str | None = None,
    dry_run: bool = False,
    workers: int = 1,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    records = load_trajectories_from_dir(traj_dir)
    if not records:
        raise FileNotFoundError(f"No normalized trajectories in {traj_dir}")
    prompt = load_elicitation_prompt(prompt_path)
    prov = get_provider(provider)

    # Resume: skip (trajectory_id, step_index) pairs already present in an
    # existing output file, same convention as run_control.py / project_steps.py.
    all_rows: list[dict] = []
    done_pairs: set[tuple[str, int]] = set()
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        if len(prev):
            all_rows = prev.to_dict(orient="records")
            done_pairs = set(
                zip(prev["trajectory_id"].astype(str), prev["step_index"].astype(int))
            )
            print(f"Resuming: {len(done_pairs)} (trajectory, step) rows already in {output_path}",
                  flush=True)

    tasks: list[tuple[TrajectoryRecord, int]] = [
        (record, step.step_index)
        for record in records
        for step in record.steps
        if (str(record.trajectory_id), step.step_index) not in done_pairs
    ]
    n_total = len(done_pairs) + len(tasks)
    print(
        f"{len(records)} trajectories, {n_total} (trajectory, step) pairs total, "
        f"{len(tasks)} pending, {workers} worker(s)",
        flush=True,
    )

    def _checkpoint() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_parquet(output_path, index=False)

    if not tasks:
        pass
    elif workers <= 1:
        for record, step_index in tasks:
            all_rows.append(
                _elicit_one_step(
                    record, step_index, prompt=prompt, api_base=api_base,
                    api_key=api_key, model=model, provider=prov, dry_run=dry_run,
                )
            )
            if len(all_rows) % checkpoint_every == 0:
                _checkpoint()
                print(f"  [{len(all_rows)}/{n_total}] checkpointed -> {output_path}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Future, tuple[str, int]] = {
                pool.submit(
                    _elicit_one_step, record, step_index, prompt=prompt, api_base=api_base,
                    api_key=api_key, model=model, provider=prov, dry_run=dry_run,
                ): (str(record.trajectory_id), step_index)
                for record, step_index in tasks
            }
            n_done = len(all_rows)
            for future in as_completed(futures):
                tid, sidx = futures[future]
                try:
                    all_rows.append(future.result())
                except Exception as exc:  # noqa: BLE001 - keep the batch alive
                    print(f"  WARN {tid} step {sidx}: worker raised "
                          f"({type(exc).__name__}: {exc})", flush=True)
                n_done += 1
                if n_done % checkpoint_every == 0:
                    _checkpoint()
                    print(f"  [{n_done}/{n_total}] checkpointed -> {output_path}", flush=True)

    _checkpoint()
    df = pd.DataFrame(all_rows)
    n_ok = int(df["p_success"].notna().sum()) if len(df) else 0
    print(
        f"Wrote {len(df)} rows ({n_ok} parsed) to {output_path}",
        flush=True,
    )
    return df


def main() -> None:
    load_dotenv()  # picks up repo-root .env (searches cwd and parent dirs)
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", type=Path, default=NORMALIZED_DIR)
    ap.add_argument("--output", type=Path, default=data_file("confidence.parquet"))
    ap.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible base URL. Defaults to https://openrouter.ai/api/v1 "
        "for --provider openrouter, else http://localhost:8000/v1 (vLLM).",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="Defaults to $OPENROUTER_API_KEY if --provider openrouter, else $MODEL_API_KEY, else EMPTY",
    )
    ap.add_argument(
        "--model",
        default=defaults["model"].split("/")[-1],
        help="Served model name (must match --served-model-name). For OpenRouter, "
        "the full slug e.g. qwen/qwen3-32b -- verify the exact id in OpenRouter's "
        "own model listing, this is not stripped like Azure's litellm-style prefixes.",
    )
    ap.add_argument("--prompt-path", type=Path, default=None)
    ap.add_argument(
        "--provider",
        default=None,
        choices=["vllm", "azure", "openrouter"],
        help="Request-shape adapter (default: MODEL_PROVIDER env or vllm)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the API; emit deterministic fake P(success) for wiring tests",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent elicitation calls (thread pool; purely I/O-bound HTTP "
        "requests). 1 = sequential, original behavior.",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write --output to disk every N completed (trajectory, step) rows.",
    )
    args = ap.parse_args()

    resolved_provider = get_provider(args.provider)

    api_base = args.api_base
    if api_base is None:
        api_base = (
            "https://openrouter.ai/api/v1"
            if resolved_provider == "openrouter"
            else "http://localhost:8000/v1"
        )

    api_key = args.api_key
    if api_key is None:
        if resolved_provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise SystemExit(
                    "OPENROUTER_API_KEY not set (checked environment and .env). "
                    "Set it or pass --api-key explicitly."
                )
        else:
            api_key = os.environ.get("MODEL_API_KEY", "EMPTY")

    run(
        args.traj_dir,
        output_path=args.output,
        api_base=api_base,
        api_key=api_key,
        model=args.model,
        prompt_path=args.prompt_path,
        provider=args.provider,
        dry_run=args.dry_run,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
