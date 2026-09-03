"""Normalized trajectory schema shared across Stage 2.

A *normalized* trajectory is the model-agnostic representation the projection
step reads: one record per agent run, one step per model turn. It is decoupled
from the raw SWE-agent ``.traj`` layout so downstream code (projection,
analyses, mock generators) depends only on this schema, not on the agent
harness that produced it.

On-disk form (one JSON file per trajectory) matches
``data/normalized_test/sample.json``::

    {
      "trajectory_id": "...",
      "outcome": 0 | 1,
      "n_steps": N,
      "steps": [
        {
          "step_index": i,
          "messages_before_gen": [{"role": ..., "content": ..., ...}, ...],
          "assistant_message": {"role": "assistant", "content": ...,
                                "reasoning_content": ..., "tool_calls": [...]},
          "assistant_response": "...",
          "observation": "..."
        },
        ...
      ]
    }

Thinking-mode note (project decision 2026-07-17: thinking is ON in Stage 2):
under Qwen3 thinking-on with vLLM's reasoning + hermes tool parsers, a generated
turn is split into three fields — ``content`` (post-think prose),
``reasoning_content`` (the ``<think>`` text), and structured ``tool_calls``. The
authoritative record of a turn is therefore the native ``assistant_message``
dict, which the projection step feeds back through the Qwen3 chat template to
reproduce the exact token stream the model generated. ``messages_before_gen`` is
likewise kept as native message dicts (assistant ``tool_calls``, ``tool``-role
``tool_call_id``) so the replayed context matches generation token-for-token.
``assistant_response`` remains a flattened convenience string (``<think>`` +
content) for analyses and human inspection; it is not used to reconstruct tokens
when ``assistant_message`` is present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A message dict may carry more than {role, content}: assistant turns add
# ``reasoning_content`` / ``tool_calls`` and tool turns add ``tool_call_id``.
Message = dict[str, Any]


@dataclass
class TrajectoryStep:
    """One model turn: the context it saw, what it produced, what came back.

    ``assistant_message`` is the native, structured assistant turn (role,
    content, reasoning_content, tool_calls) and is authoritative for token-level
    replay. It is ``None`` for legacy thinking-off trajectories, where
    ``assistant_response`` (a plain string) is the only representation.
    """

    step_index: int
    messages_before_gen: list[Message]
    assistant_response: str
    observation: str
    assistant_message: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "step_index": self.step_index,
            "messages_before_gen": self.messages_before_gen,
            "assistant_response": self.assistant_response,
            "observation": self.observation,
        }
        if self.assistant_message is not None:
            d["assistant_message"] = self.assistant_message
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(
            step_index=int(d["step_index"]),
            messages_before_gen=list(d.get("messages_before_gen", [])),
            assistant_response=d.get("assistant_response", "") or "",
            observation=d.get("observation", "") or "",
            assistant_message=d.get("assistant_message"),
        )


@dataclass
class TrajectoryRecord:
    """A full agent run plus its resolved/unresolved label.

    ``task_id`` is the SWE-bench instance (the *task*), while ``trajectory_id``
    identifies one rollout of that task (instance + seed, e.g.
    ``django__django-10097__r1``). Resampling and cross-validation are at the
    task level, so the two must stay distinct whenever several trajectories
    share one ``task_id``. ``seed`` is the rollout's
    sampling seed (``None`` for single-rollout/legacy data) and ``exit_status``
    is the agent's termination reason (mini's ``info.exit_status``), kept so the
    model-vs-infrastructure failure taxonomy is a first-class analysis filter.
    """

    trajectory_id: str
    outcome: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    n_steps: int = 0
    task_id: str = ""
    seed: int | None = None
    exit_status: str | None = None

    def __post_init__(self) -> None:
        # n_steps is derived, never trusted from the caller: keeping it in sync
        # with len(steps) means rel_pos in the projection step can't drift.
        self.n_steps = len(self.steps)
        # Legacy/single-rollout data carries no separate task id; fall back to
        # the trajectory id so grouping code always has a task key.
        if not self.task_id:
            self.task_id = self.trajectory_id

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "outcome": self.outcome,
            "exit_status": self.exit_status,
            "n_steps": self.n_steps,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryRecord":
        return cls(
            trajectory_id=str(d["trajectory_id"]),
            outcome=int(d["outcome"]),
            steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
            task_id=str(d.get("task_id") or d["trajectory_id"]),
            seed=d.get("seed"),
            exit_status=d.get("exit_status"),
        )


def save_trajectory(record: TrajectoryRecord, path: Path) -> Path:
    """Write a single normalized trajectory to ``path`` as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load_trajectory(path: Path) -> TrajectoryRecord:
    """Load a single normalized trajectory JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrajectoryRecord.from_dict(data)


def _looks_like_trajectory(data: object) -> bool:
    return (
        isinstance(data, dict)
        and "trajectory_id" in data
        and "steps" in data
        and "outcome" in data
    )


def load_trajectories_from_dir(traj_dir: Path) -> list[TrajectoryRecord]:
    """Load every normalized trajectory JSON in ``traj_dir``.

    Files that don't look like normalized trajectories (e.g. a run manifest or
    ``results.json`` dropped alongside) are skipped rather than raising, so the
    directory can double as a run folder. Returned sorted by filename for
    deterministic downstream ordering.
    """
    traj_dir = Path(traj_dir)
    records: list[TrajectoryRecord] = []
    for path in sorted(traj_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _looks_like_trajectory(data):
            continue
        records.append(TrajectoryRecord.from_dict(data))
    return records
