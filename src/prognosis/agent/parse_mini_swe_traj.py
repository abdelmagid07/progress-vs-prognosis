"""Parse a raw *mini-swe-agent* trajectory (``.traj.json``) into a normalized record.

``mini-swe-agent`` is the third-party lightweight SWE-bench agent the paper
data is generated with. Its SWE-bench batch runner writes one file per instance
at ``{output_dir}/{instance_id}/{instance_id}.traj.json``::

    {
      "info": {
        "exit_status": "Submitted" | "LimitsExceeded" | "<ExceptionName>" | ...,
        "submission":  "<patch>",
        "model_stats": {...},
        "config":      {...},
        "mini_version": "..."
      },
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<task + instructions>"},
        {"role": "assistant", "content": "<raw model turn>", "extra": {"actions": [...]}},
        {"role": "user",      "content": "<observation>",     "extra": {"returncode": ...}},
        ...
        {"role": "exit",      "content": "...", "extra": {"exit_status": ...}}   # optional trailer
      ]
    }

Unlike SWE-agent's ``.traj`` (which stores a per-step ``query`` snapshot), mini
keeps a single running ``messages`` list for the whole run. We reconstruct the
normalized per-step view from it: each assistant message is one step, everything
before it is ``messages_before_gen``, and the following user/tool message is the
``observation``. This is exactly on-policy — the running prefix *is* what the
model saw when it produced that turn.

The resolved/unresolved *outcome* is not in this file; it comes from the
SWE-bench evaluation harness ``results.json`` and is passed in by the caller
(see :mod:`prognosis.agent.ingest_batch`).

Thinking-mode note (thinking is enabled throughout): with Qwen3
thinking-on served through vLLM's reasoning + hermes tool parsers, each stored
assistant message is ``response.choices[0].message`` dumped verbatim, so its
``content`` is the post-think prose, the ``<think>`` text lives in a sibling
``reasoning_content`` field, and the tool call is structured in ``tool_calls``
(mini's own bookkeeping sits under ``extra`` and is dropped). We keep all three
native fields in ``assistant_message`` — flattening to ``content`` would silently
drop the reasoning and tool-call tokens the model actually generated. The
projection step re-renders these through the Qwen3 chat template to reproduce
the exact token stream, and ``messages_before_gen`` is kept native for the same
reason (history assistant ``tool_calls`` and ``tool``-role ``tool_call_id`` must
render identically to how they were sent at generation time).
"""

from __future__ import annotations

import json
from pathlib import Path

from prognosis.agent.parse_swe_traj import _coerce_content
from prognosis.agent.schema import Message, TrajectoryRecord, TrajectoryStep

# Roles whose message, when it immediately follows an assistant turn, is the
# environment's response to that turn (the "observation").
_OBSERVATION_ROLES = frozenset({"user", "tool"})

# The synthetic terminal marker mini appends after the last real turn. The model
# never saw it, so it must never leak into any step's context.
_SYNTHETIC_ROLES = frozenset({"exit"})


def _clean_tool_calls(tool_calls: object) -> list[dict]:
    """Normalize an OpenAI-style ``tool_calls`` list to a minimal, JSON-safe form.

    We keep only the fields the Qwen3 chat template renders (id, type, function
    name + arguments) so the re-rendered turn matches generation and the on-disk
    record stays free of provider-specific noise. ``arguments`` is preserved
    as-is (mini/litellm store it as a JSON string) — the template stringifies it.
    """
    if not isinstance(tool_calls, list):
        return []
    out: list[dict] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        out.append(
            {
                "id": tc.get("id", "") or "",
                "type": tc.get("type", "function") or "function",
                "function": {
                    "name": fn.get("name", "") or "",
                    "arguments": fn.get("arguments", "") or "",
                },
            }
        )
    return out


def _assistant_message(msg: dict) -> dict:
    """Build the native assistant turn kept for token-level replay.

    Azure Foundry Qwen deployments return the think block as ``reasoning``;
    vLLM's reasoning parser uses ``reasoning_content``. Normalize both to
    ``reasoning_content`` so projection / thinking guards stay provider-agnostic.
    """
    out: dict = {"role": "assistant", "content": _coerce_content(msg.get("content", ""))}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        out["reasoning_content"] = _coerce_content(reasoning)
    tool_calls = _clean_tool_calls(msg.get("tool_calls"))
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _derived_response(assistant_message: dict) -> str:
    """Flatten a native assistant turn to a human/analysis convenience string.

    Mirrors the flat token stream the model generated: the ``<think>`` block (if
    any) followed by the post-think content. Not used for token reconstruction
    when the native ``assistant_message`` is available.
    """
    content = assistant_message.get("content", "") or ""
    reasoning = assistant_message.get("reasoning_content")
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    return content


def _history_message(msg: dict) -> Message | None:
    """Normalize one prior-turn message for ``messages_before_gen``.

    Preserves the structure the Qwen3 template needs to reproduce the generation
    prompt: assistant ``tool_calls`` and ``tool``-role ``tool_call_id``. The
    prior-turn ``reasoning_content`` is intentionally dropped — Qwen3's template
    strips ``<think>`` blocks from history, so the model never re-saw them.
    Synthetic (``exit``) turns are skipped; provider ``extra`` is discarded.
    """
    role = str(msg.get("role", ""))
    if role in _SYNTHETIC_ROLES:
        return None
    out: Message = {"role": role, "content": _coerce_content(msg.get("content", ""))}
    if role == "assistant":
        tool_calls = _clean_tool_calls(msg.get("tool_calls"))
        if tool_calls:
            out["tool_calls"] = tool_calls
    elif role == "tool":
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            out["tool_call_id"] = tool_call_id
    return out


def mini_instance_id(info: dict, traj_path: Path) -> str:
    """Best-effort instance id for a mini trajectory.

    mini's SWE-bench batch names files ``<instance_id>.traj.json`` and does not
    always duplicate the id inside the JSON, so the filename is the reliable
    source; ``info`` / ``info.config`` are tried first in case a future version
    records it there.
    """
    for key in ("instance_id", "instance"):
        val = info.get(key)
        if val:
            return str(val)
    config = info.get("config")
    if isinstance(config, dict):
        for key in ("instance_id", "instance"):
            val = config.get(key)
            if val:
                return str(val)
    name = traj_path.name
    for suffix in (".traj.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return traj_path.stem


def load_messages(data: dict) -> list[dict]:
    """Return the running message list, tolerating minor layout differences."""
    messages = data.get("messages")
    if isinstance(messages, list):
        return messages
    # Fall back to a nested/older layout if a variant ever stores it elsewhere.
    trajectory = data.get("trajectory")
    if isinstance(trajectory, list):
        return trajectory
    return []


def _prefix_messages(messages: list) -> list[Message]:
    """Normalize a message prefix to native dicts, dropping synthetic turns."""
    out: list[Message] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        normalized = _history_message(msg)
        if normalized is not None:
            out.append(normalized)
    return out


def parse_mini_swe_traj(
    traj_path: Path,
    outcome: int,
    *,
    seed: int | None = None,
    task_id: str | None = None,
) -> TrajectoryRecord:
    """Parse one raw mini-swe-agent ``.traj.json`` into a normalized record.

    ``task_id`` defaults to the SWE-bench instance id; when a rollout ``seed`` is
    given the ``trajectory_id`` is suffixed ``__r<seed>`` so multiple seed-only
    rollouts of the same task don't collide. The reported runs are single-seed.
    """
    traj_path = Path(traj_path)
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    info = data.get("info", {}) or {}
    instance_id = task_id or mini_instance_id(info, traj_path)
    trajectory_id = f"{instance_id}__r{seed}" if seed is not None else instance_id
    exit_status = str(info.get("exit_status", "")) or None

    messages = load_messages(data)
    steps: list[TrajectoryStep] = []
    step_index = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        observation = ""
        if i + 1 < len(messages):
            nxt = messages[i + 1]
            if isinstance(nxt, dict) and str(nxt.get("role", "")) in _OBSERVATION_ROLES:
                observation = _coerce_content(nxt.get("content", ""))

        assistant_message = _assistant_message(msg)
        steps.append(
            TrajectoryStep(
                step_index=step_index,
                messages_before_gen=_prefix_messages(messages[:i]),
                assistant_response=_derived_response(assistant_message),
                observation=observation,
                assistant_message=assistant_message,
            )
        )
        step_index += 1

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        outcome=int(outcome),
        steps=steps,
        task_id=instance_id,
        seed=seed,
        exit_status=exit_status,
    )
