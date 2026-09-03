"""Parse a raw SWE-agent ``.traj`` file into a normalized :class:`TrajectoryRecord`.

SWE-agent (and the mini-agent, which mimics its output) writes one JSON file per
instance::

    {
      "trajectory": [
        {
          "response":    "<full assistant message, incl. code fence>",
          "thought":     "<reasoning>",
          "action":      "<parsed command>",
          "observation": "<tool output fed back next turn>",
          "query":       [{"role": ..., "content": ...}, ...]  # context for THIS turn
        },
        ...
      ],
      "info": {"instance_id": "...", "exit_status": "..."}
    }

``query`` is the message list the model saw before generating ``response`` for
that step, which is exactly ``messages_before_gen`` in the normalized schema.

The resolved/unresolved *outcome* is not in the ``.traj`` file — it comes from
the SWE-bench ``results.json`` and is passed in by the caller (see
``ingest_batch``).
"""

from __future__ import annotations

import json
from pathlib import Path

from prognosis.agent.schema import (
    Message,
    TrajectoryRecord,
    TrajectoryStep,
)


def _coerce_content(content: object) -> str:
    """Flatten a message ``content`` to a string.

    Most turns carry a plain string. Some SWE-agent / OpenAI-style messages use
    a list of typed blocks (``[{"type": "text", "text": ...}, ...]``); join the
    text of those so a real trajectory doesn't break the projection step.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _normalize_messages(query: object) -> list[Message]:
    if not isinstance(query, list):
        return []
    messages: list[Message] = []
    for msg in query:
        if not isinstance(msg, dict):
            continue
        messages.append(
            {
                "role": str(msg.get("role", "")),
                "content": _coerce_content(msg.get("content", "")),
            }
        )
    return messages


def parse_swe_traj(
    traj_path: Path,
    outcome: int,
    *,
    seed: int | None = None,
    task_id: str | None = None,
) -> TrajectoryRecord:
    """Parse one raw ``.traj`` file into a normalized record with ``outcome``.

    Mirrors :func:`parse_mini_swe_traj`'s identity handling: ``task_id`` defaults
    to the instance id and a rollout ``seed`` suffixes the ``trajectory_id``.
    """
    traj_path = Path(traj_path)
    data = json.loads(traj_path.read_text(encoding="utf-8"))

    raw_steps = data.get("trajectory", [])
    info = data.get("info", {}) or {}
    instance_id = task_id or str(info.get("instance_id") or traj_path.stem)
    trajectory_id = f"{instance_id}__r{seed}" if seed is not None else instance_id
    exit_status = str(info.get("exit_status", "")) or None

    steps: list[TrajectoryStep] = []
    for i, raw in enumerate(raw_steps):
        steps.append(
            TrajectoryStep(
                step_index=i,
                messages_before_gen=_normalize_messages(raw.get("query")),
                assistant_response=_coerce_content(raw.get("response", "")),
                observation=_coerce_content(raw.get("observation", "")),
            )
        )

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        outcome=int(outcome),
        steps=steps,
        task_id=instance_id,
        seed=seed,
        exit_status=exit_status,
    )
