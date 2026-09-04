#!/usr/bin/env python
"""Ingest a batch of raw agent trajectories into normalized trajectories.

Reads a run directory as produced by the ``mini-swe-agent`` SWE-bench batch
runner, labels each trajectory from the run's ``results.json``, and writes one normalized JSON per trajectory into the
directory the projection step consumes.

    python -m prognosis.agent.ingest_batch --traj-dir data/trajectories/run_<ts>

Two raw formats are supported via ``--format``:

* ``mini-swe-agent`` (default): ``*.traj.json`` with a running ``messages`` list.
* ``swe-agent``: ``*.traj`` with a per-step ``trajectory``/``query`` layout.

Labels come from ``results.json`` (``resolved_ids`` / ``unresolved_ids``, or the
SWE-bench harness ``resolved`` / ``unresolved`` keys). A trajectory whose
instance id is in neither list is skipped with a warning: the transfer test is
supervised, so an unlabeled trajectory is unusable.

Reproduce: python -m prognosis.agent.ingest_batch --traj-dir data/trajectories/<run>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from prognosis.common.paths import NORMALIZED_DIR
from prognosis.agent.parse_mini_swe_traj import mini_instance_id, parse_mini_swe_traj
from prognosis.agent.parse_swe_traj import parse_swe_traj
from prognosis.agent.schema import TrajectoryRecord, save_trajectory

_RESOLVED_KEYS = ("resolved_ids", "resolved")
_UNRESOLVED_KEYS = ("unresolved_ids", "unresolved")
# SWE-bench harness reports empty patches in a third bucket (never applied /
# never tested). For this project they are model failures → unresolved.
_EMPTY_PATCH_KEYS = ("empty_patch_ids", "empty_patch")


def _first_present(payload: dict, keys: tuple[str, ...]) -> tuple[set[str], bool]:
    """Return (ids, key_was_present) for the first matching *list/set* key.

    Skips non-sequence values so integer summary fields like
    ``unresolved_instances: 1`` in harness reports are ignored.
    """
    for key in keys:
        if key not in payload:
            continue
        val = payload[key]
        if isinstance(val, (list, set, tuple)):
            return set(val or []), True
    return set(), False


def load_labels(results_path: Path) -> tuple[dict[str, int], bool]:
    """Load instance_id -> outcome from a results.json.

    Returns the label map and whether an explicit unresolved list was present.
    When only a resolved list exists (common in SWE-bench reports), the caller
    treats any other instance as unresolved rather than dropping it.

    Empty-patch ids from the harness (``empty_patch_ids``) are mapped to
    unresolved (outcome 0): no patch means the agent did not resolve the task.
    """
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    resolved, _ = _first_present(payload, _RESOLVED_KEYS)
    unresolved, had_unresolved = _first_present(payload, _UNRESOLVED_KEYS)
    empty_patch, had_empty = _first_present(payload, _EMPTY_PATCH_KEYS)

    labels = {iid: 1 for iid in resolved}
    for iid in unresolved | empty_patch:
        labels.setdefault(iid, 0)
    # empty_patch counts as an explicit failure label even if unresolved_ids
    # omitted them (harness third bucket).
    return labels, had_unresolved or had_empty


# --- Crashed-run detection ------------------------------------------------
# A "crashed" run failed for infrastructure reasons (API/transport error, dead
# container) rather than because the model tried the task and failed. Its
# trajectory is a stub, and counting it as a task failure poisons the outcome
# labels, so we exclude it before labeling.
#
# The two agents encode this differently:
#   swe-agent  -> a single sentinel exit_status ("error"); use a DENYLIST.
#   mini       -> genuine terminations are a small fixed set, while crashes take
#                 the *exception class name* (e.g. "APIConnectionError"), which
#                 is open-ended; so we ALLOWLIST the genuine ones and treat
#                 everything else as a crash.
_SWEAGENT_ERROR_STATUSES = frozenset({"error"})
_MINI_GENUINE_STATUSES = frozenset(
    {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}
)


@dataclass(frozen=True)
class FormatSpec:
    """How to read one raw trajectory format."""

    name: str
    glob: str
    parse: Callable[..., TrajectoryRecord]
    instance_id: Callable[[dict, Path], str]
    is_crash: Callable[[str], bool]


_SEED_DIR_RE = re.compile(r"^r(\d+)$")


def find_results_for(traj_path: Path, traj_dir: Path) -> Path | None:
    """Nearest ``results.json`` for a trajectory, walking parents up to ``traj_dir``.

    Outcomes are per-rollout, so a multi-rollout run has one ``results.json`` per
    ``r<seed>/`` subdir; a flat single-rollout run has one at the top. Walking up
    from the trajectory file finds the right one in both layouts (and lets a
    single top-level report still cover a flat run).
    """
    traj_dir_r = Path(traj_dir).resolve()
    cur = Path(traj_path).resolve().parent
    while True:
        cand = cur / "results.json"
        if cand.exists():
            return cand
        if cur == traj_dir_r:
            return None
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent


def _seed_from_path(traj_path: Path, traj_dir: Path) -> int | None:
    """Recover a rollout seed from an ``r<seed>/`` ancestor directory.

    The multi-rollout runner writes each seed's trajectories under
    ``<run>/r<seed>/…``. A flat run directory (single rollout)
    has no such component and yields ``None``.
    """
    try:
        rel = traj_path.relative_to(traj_dir)
    except ValueError:
        rel = traj_path
    for part in rel.parts:
        m = _SEED_DIR_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def _sweagent_instance_id(info: dict, path: Path) -> str:
    return str(info.get("instance_id") or path.stem)


def _mini_is_crash(exit_status: str, genuine: frozenset[str]) -> bool:
    status = (exit_status or "").strip()
    if not status:
        return False  # unknown; keep it and let the histogram surface it
    if status.startswith("Submitted"):
        return False  # e.g. "Submitted" / "Submitted (exit_command)"
    return status not in genuine


def build_format_spec(
    fmt: str,
    *,
    error_statuses: frozenset[str],
    genuine_statuses: frozenset[str],
) -> FormatSpec:
    if fmt == "mini-swe-agent":
        return FormatSpec(
            name=fmt,
            glob="*.traj.json",
            parse=parse_mini_swe_traj,
            instance_id=mini_instance_id,
            is_crash=lambda s: _mini_is_crash(s, genuine_statuses),
        )
    if fmt == "swe-agent":
        return FormatSpec(
            name=fmt,
            glob="*.traj",
            parse=parse_swe_traj,
            instance_id=_sweagent_instance_id,
            is_crash=lambda s: (s or "") in error_statuses,
        )
    raise SystemExit(f"Unknown --format {fmt!r}; expected mini-swe-agent or swe-agent")


def find_traj_files(traj_dir: Path, glob: str) -> list[Path]:
    """All raw trajectory files under ``traj_dir`` (agents nest them per
    instance), sorted for deterministic ordering."""
    return sorted(Path(traj_dir).rglob(glob))


def ingest_batch(
    traj_dir: Path,
    *,
    fmt: str = "mini-swe-agent",
    results_path: Path | None = None,
    output_dir: Path = NORMALIZED_DIR,
    error_statuses: frozenset[str] = _SWEAGENT_ERROR_STATUSES,
    genuine_statuses: frozenset[str] = _MINI_GENUINE_STATUSES,
    keep_error_stubs: bool = False,
) -> dict:
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        raise SystemExit(f"Trajectory directory not found: {traj_dir}")

    spec = build_format_spec(
        fmt, error_statuses=error_statuses, genuine_statuses=genuine_statuses
    )

    # Labels can be a single shared report (--results, or a flat run's
    # results.json) or one report per rollout subdir. Resolve lazily per file so
    # both layouts work; cache each report so it is parsed once.
    explicit_results = Path(results_path) if results_path else None
    if explicit_results is not None and not explicit_results.exists():
        raise SystemExit(f"Labels file not found: {explicit_results}")
    label_cache: dict[Path, tuple[dict[str, int], bool]] = {}

    def labels_for(tp: Path) -> tuple[dict[str, int] | None, bool, Path | None]:
        rp = explicit_results or find_results_for(tp, traj_dir)
        if rp is None:
            return None, False, None
        if rp not in label_cache:
            label_cache[rp] = load_labels(rp)
        labels, had_unresolved = label_cache[rp]
        return labels, had_unresolved, rp

    traj_files = find_traj_files(traj_dir, spec.glob)
    if not traj_files:
        raise SystemExit(
            f"No {spec.glob} files found under {traj_dir} (format={spec.name})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    skipped: list[str] = []
    excluded_errors: list[dict] = []
    exit_status_hist: Counter[str] = Counter()
    results_paths_used: set[str] = set()
    n_success = 0
    n_failure = 0

    for traj_path in traj_files:
        # Peek at info to resolve label and exit status before a full parse.
        info = json.loads(traj_path.read_text(encoding="utf-8")).get("info", {}) or {}
        instance_id = spec.instance_id(info, traj_path)

        exit_status = str(info.get("exit_status", ""))
        exit_status_hist[exit_status or "(none)"] += 1

        if not keep_error_stubs and spec.is_crash(exit_status):
            # Crashed run, not a real task failure: drop before labeling.
            print(
                f"  EXCLUDE {instance_id}: crashed run "
                f"(exit_status={exit_status!r}), not a task failure",
                flush=True,
            )
            excluded_errors.append(
                {
                    "trajectory_id": instance_id,
                    "task_id": instance_id,
                    "seed": _seed_from_path(traj_path, traj_dir),
                    "exit_status": exit_status,
                    "source": str(traj_path),
                }
            )
            continue

        # Labels are keyed by the SWE-bench *task* (instance) id within each
        # rollout's own report (outcomes differ across rollouts of a task).
        labels, had_unresolved, rp = labels_for(traj_path)
        if labels is None:
            print(
                f"  SKIP {instance_id}: no results.json found for its rollout "
                "(evaluate the rollout's preds.json with the harness first)",
                flush=True,
            )
            skipped.append(instance_id)
            continue
        if rp is not None:
            results_paths_used.add(str(rp))
        if instance_id in labels:
            outcome = labels[instance_id]
        elif not had_unresolved:
            # Only a resolved list was given: absence means unresolved.
            outcome = 0
        else:
            print(f"  SKIP {instance_id}: no label in {rp}", flush=True)
            skipped.append(instance_id)
            continue

        seed = _seed_from_path(traj_path, traj_dir)
        record = spec.parse(traj_path, outcome, seed=seed, task_id=instance_id)
        out_path = output_dir / f"{record.trajectory_id}.json"
        save_trajectory(record, out_path)
        written.append(
            {
                "trajectory_id": record.trajectory_id,
                "task_id": record.task_id,
                "seed": record.seed,
                "outcome": outcome,
                "exit_status": record.exit_status,
                "n_steps": record.n_steps,
                "source": str(traj_path),
            }
        )
        n_success += int(outcome == 1)
        n_failure += int(outcome == 0)
        print(
            f"  {record.trajectory_id}: {record.n_steps} steps, "
            f"outcome={outcome} -> {out_path.name}",
            flush=True,
        )

    if not results_paths_used and not written:
        raise SystemExit(
            f"No results.json found under {traj_dir} (nor via --results).\n"
            "Ingest needs resolved/unresolved labels. For multi-rollout runs, "
            "evaluate each r<seed>/preds.json and place its report at "
            "r<seed>/results.json; for a flat run, place one at the run root."
        )

    manifest = {
        "format": spec.name,
        "traj_dir": str(traj_dir),
        "results_paths": sorted(results_paths_used),
        "output_dir": str(output_dir),
        "n_ingested": len(written),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_skipped": len(skipped),
        "n_excluded_errors": len(excluded_errors),
        "exit_status_histogram": dict(sorted(exit_status_hist.items())),
        "skipped": skipped,
        "excluded_errors": excluded_errors,
        "trajectories": written,
    }
    manifest_path = output_dir / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nIngested {len(written)} trajectories "
        f"(N_success={n_success}, N_failure={n_failure}, "
        f"skipped={len(skipped)}, excluded_errors={len(excluded_errors)}) "
        f"-> {output_dir}",
        flush=True,
    )
    # The histogram is the safety net for the crash guard: if a genuine
    # termination is being excluded (or a crash slipping through), it shows here.
    print("  exit_status histogram:", flush=True)
    for status, count in sorted(exit_status_hist.items()):
        print(f"    {status}: {count}", flush=True)
    if n_success == 0 or n_failure == 0:
        print(
            "  WARNING: only one outcome class present; "
            "expand the instance list before running the transfer analysis.",
            flush=True,
        )
    print(
        f"Next: python -m prognosis.extract.project_steps --traj-dir {output_dir}",
        flush=True,
    )
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traj-dir",
        type=Path,
        required=True,
        help="Run directory containing raw trajectory files and results.json",
    )
    ap.add_argument(
        "--format",
        choices=["mini-swe-agent", "swe-agent"],
        default="mini-swe-agent",
        help="Raw trajectory format to ingest (default: mini-swe-agent)",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Labels file (default: <traj-dir>/results.json)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=NORMALIZED_DIR,
        help="Where to write normalized trajectory JSON (default: data/normalized)",
    )
    ap.add_argument(
        "--error-statuses",
        nargs="+",
        default=sorted(_SWEAGENT_ERROR_STATUSES),
        help="[swe-agent] exit_status values to exclude as crashed runs "
        "(default: error).",
    )
    ap.add_argument(
        "--genuine-statuses",
        nargs="+",
        default=sorted(_MINI_GENUINE_STATUSES),
        help="[mini-swe-agent] exit_status values that mean a genuine task "
        "attempt; anything else (an exception class name) is treated as a "
        "crashed run and excluded. Inspect the ingest histogram and extend if "
        "a new genuine status appears.",
    )
    ap.add_argument(
        "--keep-error-stubs",
        action="store_true",
        help="Ingest crashed-run stubs instead of excluding them (not recommended: "
        "they corrupt the outcome labels)",
    )
    args = ap.parse_args()
    ingest_batch(
        args.traj_dir,
        fmt=args.format,
        results_path=args.results,
        output_dir=args.output_dir,
        error_statuses=frozenset(args.error_statuses),
        genuine_statuses=frozenset(args.genuine_statuses),
        keep_error_stubs=args.keep_error_stubs,
    )


if __name__ == "__main__":
    main()
