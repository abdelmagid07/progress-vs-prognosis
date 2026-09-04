#!/usr/bin/env python
"""Generate a FAILING arm for each task in the pair instance list.

Every task in data/pair_instances.txt already has a resolved trajectory in the main run.
A within-task pair needs the other arm, so this resamples each task --
same prompt, same sampling settings, fresh seed -- until it produces an UNRESOLVED trajectory.

Why a driver instead of one batch call: whether an attempt is usable is only known after the
SWE-bench harness evaluates it, so the loop is generate -> evaluate -> keep the failures ->
retry the rest. Doing that by hand across dozens of tasks and several rounds is where mistakes
get made (a re-used seed, a round evaluated against the wrong preds.json, a task quietly dropped).

Requires Docker, mini-swe-agent (`mini-extra`), and the swebench harness: the same
environment that produced the main run.

State lives in OUTPUT_DIR/pair_state.json and every step is resumable: re-run the
same command after an interruption and it picks up with only the tasks still missing a failure.

    # dry run first -- prints the plan, touches nothing
    python scripts/generate_pair_arms.py --dry-run

    export MODEL_PROVIDER=azure MODEL_API_BASE=... MODEL_API_KEY=... MODEL_NAME=openai/qwen3-32b
    python scripts/generate_pair_arms.py --workers 8

The provider MUST match whatever generated the main run, or the two arms of a pair differ by
more than sampling and the contrast is no longer clean. See --require-provider.

Reproduce: python scripts/generate_pair_arms.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "pair_instances.txt"
RUNNER = ROOT / "scripts" / "run_mini_swe_batch.sh"
DATASET = "princeton-nlp/SWE-bench_Verified"

# Seeds start well above anything the main run used so a "fresh seed" is fresh by construction
# rather than by our memory of what was used before.
SEED_START = 1000


def read_instances(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun: python scripts/select_pair_tasks.py")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    if not out:
        raise SystemExit(f"{path} lists no instance ids")
    return out


def load_state(path: Path, tasks: list[str]) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        # A task list edited between runs would silently strand tasks; catch it loudly.
        known = set(state["tasks"])
        if known != set(tasks):
            raise SystemExit(
                f"{path} was built from a different task list "
                f"({len(known)} tasks vs {len(tasks)} now). Delete it to start over, "
                f"or restore the original pair_instances.txt.")
        return state
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": {t: {"attempts": [], "failing_seed": None} for t in tasks},
        "rounds": [],
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: an interrupt mid-write must not corrupt the resume point


def pending(state: dict, max_attempts: int) -> list[str]:
    return sorted(
        t for t, rec in state["tasks"].items()
        if rec["failing_seed"] is None and len(rec["attempts"]) < max_attempts
    )


def exhausted(state: dict, max_attempts: int) -> list[str]:
    return sorted(
        t for t, rec in state["tasks"].items()
        if rec["failing_seed"] is None and len(rec["attempts"]) >= max_attempts
    )


def run_generation(round_dir: Path, task_file: Path, seed: int, args) -> None:
    env = os.environ.copy()
    env.update({
        "OUTPUT_DIR": str(round_dir),
        "SEED_BASE": str(seed),   # explicit seed => runner nests under r<seed>/ and injects it
        "ROLLOUTS": "1",
        "WORKERS": str(args.workers),
        "STEP_LIMIT": str(args.step_limit),
    })
    subprocess.check_call(["bash", str(RUNNER), str(task_file)], cwd=str(ROOT), env=env)


def evaluate(run_dir: Path, run_id: str, args) -> Path:
    """Evaluate one rollout's predictions and return the path to its results.json.

    The harness writes its report into the CWD as <model>.<run_id>.json rather than next to
    the predictions, so we locate it by run_id and move it into place. ingest_batch reads
    results.json from beside preds.json.
    """
    preds = run_dir / "preds.json"
    if not preds.exists():
        raise SystemExit(f"no predictions at {preds} -- generation produced nothing")

    results = run_dir / "results.json"
    if results.exists():
        return results  # already evaluated (resumed run)

    before = set(ROOT.glob(f"*.{run_id}.json"))
    subprocess.check_call(
        [sys.executable, "-m", "swebench.harness.run_evaluation",
         "--dataset_name", DATASET,
         "--predictions_path", str(preds),
         "--run_id", run_id,
         "--max_workers", str(args.workers)],
        cwd=str(ROOT),
    )
    reports = sorted(set(ROOT.glob(f"*.{run_id}.json")) - before)
    if not reports:
        reports = sorted(ROOT.glob(f"*.{run_id}.json"))
    if not reports:
        raise SystemExit(
            f"harness produced no report matching *.{run_id}.json in {ROOT}. "
            f"Evaluate by hand and place the report at {results}.")
    shutil.move(str(reports[-1]), str(results))
    return results


def read_outcomes(results: Path) -> tuple[set[str], set[str]]:
    """(resolved, unresolved) instance ids. Empty patches count as unresolved: a model that
    submits nothing has failed the task, which is the project's standing convention."""
    payload = json.loads(results.read_text(encoding="utf-8"))

    def pick(keys):
        for k in keys:
            v = payload.get(k)
            if isinstance(v, (list, tuple, set)):
                return set(v)
        return set()

    resolved = pick(("resolved_ids", "resolved"))
    unresolved = pick(("unresolved_ids", "unresolved")) | pick(("empty_patch_ids", "empty_patch"))
    return resolved, unresolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=Path, default=INSTANCES)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "trajectories" / "pairs")
    ap.add_argument("--max-attempts", type=int, default=6,
                    help="per-task generation attempts before giving up on that task")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step-limit", type=int, default=60,
                    help="must match the main run's step budget")
    ap.add_argument("--require-provider", default=None,
                    help="abort unless MODEL_PROVIDER equals this (guards against pairing "
                         "arms generated on different backends)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = read_instances(args.instances)
    out = args.output_dir
    state_path = out / "pair_state.json"
    state = load_state(state_path, tasks)

    provider = os.environ.get("MODEL_PROVIDER", "vllm")
    if args.require_provider and provider != args.require_provider:
        raise SystemExit(
            f"MODEL_PROVIDER={provider!r} but --require-provider={args.require_provider!r}. "
            f"Both arms of a pair must come from the same backend.")

    todo = pending(state, args.max_attempts)
    done = [t for t, r in state["tasks"].items() if r["failing_seed"] is not None]
    print(f"tasks            : {len(tasks)}")
    print(f"already have fail: {len(done)}")
    print(f"still needed     : {len(todo)}")
    print(f"provider         : {provider}")
    print(f"step limit       : {args.step_limit}   workers: {args.workers}")
    print(f"output           : {out}")
    if args.dry_run:
        print("\n[dry run] would generate for:")
        for t in todo:
            print("  ", t, f"(attempts so far: {len(state['tasks'][t]['attempts'])})")
        return
    if not todo:
        print("\nNothing to do. Next: python scripts/package_pairs.py")
        return

    out.mkdir(parents=True, exist_ok=True)

    for rnd in range(args.max_rounds):
        todo = pending(state, args.max_attempts)
        if not todo:
            break
        seed = SEED_START + len(state["rounds"])
        round_dir = out / f"round{len(state['rounds']):02d}_seed{seed}"
        task_file = round_dir / "instances.txt"
        round_dir.mkdir(parents=True, exist_ok=True)
        task_file.write_text("\n".join(todo) + "\n", encoding="utf-8")

        print(f"\n{'='*72}\nROUND {len(state['rounds'])}  seed={seed}  tasks={len(todo)}\n{'='*72}",
              flush=True)
        run_generation(round_dir, task_file, seed, args)

        # The runner nests an explicitly-seeded rollout under r<seed>/.
        rollout_dir = round_dir / f"r{seed}"
        if not rollout_dir.exists():
            rollout_dir = round_dir
        run_id = f"pairexp_r{seed}"
        results = evaluate(rollout_dir, run_id, args)
        resolved, unresolved = read_outcomes(results)

        newly = []
        for t in todo:
            if t in unresolved:
                state["tasks"][t]["failing_seed"] = seed
                state["tasks"][t]["attempts"].append({"seed": seed, "outcome": "unresolved"})
                newly.append(t)
            elif t in resolved:
                state["tasks"][t]["attempts"].append({"seed": seed, "outcome": "resolved"})
            else:
                # Neither list: infra failure or an unlabeled instance. Costs an attempt but
                # is not evidence about the model, so it is recorded distinctly.
                state["tasks"][t]["attempts"].append({"seed": seed, "outcome": "unlabeled"})

        state["rounds"].append({
            "seed": seed, "dir": str(round_dir.relative_to(ROOT)),
            "n_attempted": len(todo), "n_failing": len(newly),
            "n_resolved_again": len(todo) - len(newly),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        save_state(state_path, state)

        have = sum(1 for r in state["tasks"].values() if r["failing_seed"])
        print(f"\nround {len(state['rounds'])-1}: {len(newly)} new failing arms "
              f"| total {have}/{len(tasks)}", flush=True)

    have = sorted(t for t, r in state["tasks"].items() if r["failing_seed"])
    stuck = exhausted(state, args.max_attempts)
    print(f"\n{'='*72}")
    print(f"failing arms obtained : {len(have)}/{len(tasks)}")
    print(f"attempts exhausted    : {len(stuck)}")
    if stuck:
        print("  these tasks resolved on every attempt and cannot be paired without more"
              " rounds (raise --max-attempts) -- they are NOT a pipeline error:")
        for t in stuck:
            print("   ", t)
    print(f"\npairs with both arms : {len(have)}")
    print("\nNext: python scripts/package_pairs.py")


if __name__ == "__main__":
    main()
