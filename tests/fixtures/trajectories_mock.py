"""Synthetic trajectories for offline wiring tests."""

from __future__ import annotations

import json
from pathlib import Path

from prognosis.agent.parse_swe_traj import parse_swe_traj
from prognosis.agent.schema import TrajectoryRecord, TrajectoryStep, save_trajectory


def _make_record(
    trajectory_id: str,
    outcome: int,
    n_steps: int,
    *,
    task_id: str | None = None,
    seed: int | None = None,
) -> TrajectoryRecord:
    steps = []
    for i in range(n_steps):
        assistant_message = {
            "role": "assistant",
            "content": f"Step {i}: inspect and patch.",
            "reasoning_content": f"Reasoning for step {i}: I should inspect file_{i}.py.",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": f'{{"command": "cat file_{i}.py"}}'},
                }
            ],
        }
        steps.append(
            TrajectoryStep(
                step_index=i,
                messages_before_gen=[
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": f"ISSUE: fix bug {trajectory_id} (step {i})"},
                ],
                assistant_response=(
                    f"<think>\nReasoning for step {i}: I should inspect file_{i}.py.\n</think>\n\n"
                    f"Step {i}: inspect and patch."
                ),
                observation=f"file_{i}.py\nline {i * 10}: def foo(): pass\n",
                assistant_message=assistant_message,
            )
        )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        outcome=outcome,
        n_steps=n_steps,
        steps=steps,
        task_id=task_id or trajectory_id,
        seed=seed,
        exit_status="Submitted" if outcome == 1 else "LimitsExceeded",
    )


def _mini_traj(instance_id: str, exit_status: str, n_assistant: int) -> dict:
    """A minimal mini-swe-agent trajectory: system + user, then alternating
    assistant/tool turns, then the synthetic exit marker.

    Assistant turns carry the thinking-on shape (post-think ``content`` +
    ``reasoning_content`` + structured ``tool_calls``); observations are
    ``tool``-role messages keyed by ``tool_call_id``, matching the reasoning +
    hermes parser output the real runner records."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": f"ISSUE: fix bug in {instance_id}"},
    ]
    for i in range(n_assistant):
        call_id = f"call_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": f"Inspecting step {i}.",
                "reasoning_content": f"For step {i} I should list step_{i}.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": f'{{"command": "ls step_{i}"}}'},
                    }
                ],
                "extra": {"actions": [f"ls step_{i}"]},
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"<returncode>0</returncode>\n<output>\nstep_{i}.py\n</output>",
            }
        )
    messages.append({"role": "exit", "content": exit_status, "extra": {"exit_status": exit_status}})
    return {
        "info": {"instance_id": instance_id, "exit_status": exit_status, "submission": ""},
        "messages": messages,
    }


def write_mini_run_fixture(run_dir: Path) -> Path:
    """Write a tiny mini-swe-agent SWE-bench run directory for ingest tests.

    Layout mirrors the real batch runner: ``<run>/<instance_id>/<instance_id>.traj.json``
    plus a ``results.json``. Includes one resolved, one unresolved, and one
    crashed stub (an exception-class exit_status) so the ingest crash guard has
    something to exclude.
    """
    run_dir = Path(run_dir)
    specs = [
        ("mini_ok_1", "Submitted", 3),
        ("mini_bad_2", "LimitsExceeded", 4),
        ("mini_crash_3", "APIConnectionError", 1),  # infra crash -> must be excluded
    ]
    for instance_id, exit_status, n_assistant in specs:
        inst_dir = run_dir / instance_id
        inst_dir.mkdir(parents=True, exist_ok=True)
        payload = _mini_traj(instance_id, exit_status, n_assistant)
        (inst_dir / f"{instance_id}.traj.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    results = {"resolved_ids": ["mini_ok_1"], "unresolved_ids": ["mini_bad_2"]}
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results_path


def write_mini_rollout_fixture(run_dir: Path) -> Path:
    """Write a multi-rollout mini-swe-agent run: ``<run>/r<seed>/<inst>/…``.

    Two seeds over two tasks, with a per-rollout results.json so outcomes differ
    across seeds (task ``roll_a`` resolves at seed 0 but not seed 1), plus one
    infra crash at seed 1 the ingest guard must exclude and the regen planner
    must pick up. Returns the run directory.
    """
    run_dir = Path(run_dir)
    layout = {
        0: [("roll_a", "Submitted", 2), ("roll_b", "LimitsExceeded", 3)],
        1: [("roll_a", "LimitsExceeded", 2), ("roll_b", "APIConnectionError", 1)],
    }
    results = {
        0: {"resolved_ids": ["roll_a"], "unresolved_ids": ["roll_b"]},
        1: {"resolved_ids": [], "unresolved_ids": ["roll_a"]},
    }
    for seed, specs in layout.items():
        rdir = run_dir / f"r{seed}"
        for instance_id, exit_status, n_assistant in specs:
            inst_dir = rdir / instance_id
            inst_dir.mkdir(parents=True, exist_ok=True)
            payload = _mini_traj(instance_id, exit_status, n_assistant)
            (inst_dir / f"{instance_id}.traj.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        (rdir / "results.json").write_text(
            json.dumps(results[seed], indent=2), encoding="utf-8"
        )
    return run_dir


def write_smoke_fixtures(output_dir: Path, sample_traj: Path | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if sample_traj and sample_traj.exists():
        record = parse_swe_traj(sample_traj, outcome=1)
        path = output_dir / f"{record.trajectory_id}.json"
        save_trajectory(record, path)
        written.append(path)

    for tid, outcome, n_steps in [
        ("mock_success_a", 1, 4),
        ("mock_failure_b", 0, 5),
        ("mock_success_c", 1, 3),
    ]:
        record = _make_record(tid, outcome, n_steps)
        path = output_dir / f"{tid}.json"
        save_trajectory(record, path)
        written.append(path)

    manifest = {"n_trajectories": len(written), "paths": [str(p) for p in written]}
    with open(output_dir.parent / "smoke_fixtures.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return written
