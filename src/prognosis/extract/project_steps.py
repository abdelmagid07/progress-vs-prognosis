#!/usr/bin/env python
"""Replay SWE-agent trajectories through Qwen3-8B and extract value-axis projections."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prognosis.common.chat import apply_chat_template
from prognosis.extract.hooks import LayerActivationCapture, cosine_projection, unit_direction
from prognosis.common.config import load_defaults
from prognosis.common.paths import NORMALIZED_DIR, data_file, require_axis_path
from prognosis.extract.token_spans import generated_token_indices
from prognosis.agent.schema import TrajectoryRecord, load_trajectories_from_dir


def rel_pos(step_index: int, n_steps: int) -> float:
    if n_steps <= 1:
        return 0.0
    return step_index / (n_steps - 1)


_THINK_RE = re.compile(r"<think>", re.IGNORECASE)


def trajectory_uses_thinking(records: list[TrajectoryRecord]) -> bool:
    """Detect whether the trajectories were generated with thinking mode ON.

    Under Qwen3 thinking-on served with a reasoning parser, the ``<think>`` text
    is split into the native ``assistant_message.reasoning_content`` field rather
    than left inline in ``content``; older/no-parser captures leave it inline. We
    treat a trajectory as thinking-on if *any* assistant turn carries a
    ``reasoning_content`` or an inline ``<think>`` block, checking both the
    per-step ``assistant_message`` / ``assistant_response`` and the assistant
    turns in ``messages_before_gen``.
    """
    for rec in records:
        for step in rec.steps:
            if step.assistant_message and step.assistant_message.get("reasoning_content"):
                return True
            if _THINK_RE.search(step.assistant_response or ""):
                return True
            for msg in step.messages_before_gen:
                if msg.get("role") == "assistant" and (
                    msg.get("reasoning_content")
                    or msg.get("reasoning")  # Azure Foundry field name
                    or _THINK_RE.search(msg.get("content", "") or "")
                ):
                    return True
    return False


def assert_thinking_mode_matches(
    records: list[TrajectoryRecord], enable_thinking: bool
) -> None:
    """Fail loudly on a generation/projection thinking-mode mismatch.

    The projection replays each recorded turn through the chat template and reads
    the activation at its last token. If the template's thinking mode differs from
    the mode the trajectory was *generated* under, the tokenization (and thus the
    activation we read) no longer corresponds to what the model actually computed.
    That silently corrupts every projection, so we refuse to proceed either way.
    (Stage 2 runs thinking-ON by default as of 2026-07-17; legacy thinking-off
    data can still be projected with ``--no-enable-thinking``.)
    """
    generated_with_thinking = trajectory_uses_thinking(records)
    if generated_with_thinking and not enable_thinking:
        raise SystemExit(
            "Thinking-mode mismatch: trajectories were generated with thinking ON "
            "(reasoning_content / <think> blocks present) but projection is running "
            "with thinking OFF. Re-run with --enable-thinking so the chat template "
            "matches how the model actually generated."
        )
    if enable_thinking and not generated_with_thinking:
        raise SystemExit(
            "Thinking-mode mismatch: projection is running with thinking ON but the "
            "trajectories carry no reasoning_content / <think> blocks (generated "
            "thinking OFF). Re-run with --no-enable-thinking to match, or regenerate "
            "the trajectories with thinking ON."
        )


def _final_assistant_segment(full_text: str) -> str:
    """The rendered text of the last assistant turn (marker-delimited)."""
    marker = "<|im_start|>assistant"
    marker_pos = full_text.rfind(marker)
    if marker_pos < 0:
        return ""
    region_start = marker_pos + len(marker)
    end_pos = full_text.find("<|im_end|>", region_start)
    return full_text[region_start : end_pos if end_pos >= 0 else len(full_text)]


def assert_render_fidelity(full_text: str, assistant_message: dict) -> None:
    """Confirm the chat template re-rendered the native turn faithfully.

    The projection's correctness hinges on the Qwen3 template reproducing the
    generated token stream from the split ``reasoning_content`` / ``tool_calls``
    fields. If a template version drops the ``<think>`` block or mangles the tool
    call, the activation we read no longer matches what the model computed, so we
    fail loudly (the thinking-ON analogue of the old thinking-off guard). Run once
    per projection on the first structured turn.
    """
    segment = _final_assistant_segment(full_text)

    reasoning = (assistant_message.get("reasoning_content") or "").strip()
    if reasoning:
        probe = reasoning[: min(40, len(reasoning))]
        if probe and probe not in segment:
            raise SystemExit(
                "Render fidelity check failed: the assistant's reasoning_content is "
                "absent from the re-rendered turn. The Qwen3 chat template is not "
                "emitting the <think> block for completed turns, so the replayed "
                "tokens do not match generation. Verify the transformers/template "
                "version renders reasoning_content before trusting projections."
            )

    for tc in assistant_message.get("tool_calls") or []:
        name = (tc.get("function") or {}).get("name", "")
        if name and name not in segment:
            raise SystemExit(
                "Render fidelity check failed: tool call "
                f"{name!r} is absent from the re-rendered turn. The chat template is "
                "not reproducing tool_calls, so the replayed tokens do not match "
                "generation. Verify the template renders tool_calls (hermes-style) "
                "before trusting projections."
            )


def _encode_template(tokenizer, messages, enable_thinking: bool):
    text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    return text, enc["input_ids"], offset_mapping


def load_axis_directions(
    axis_path: Path, layers: list[int], axis_layer_offset: int = 0
) -> dict[int, np.ndarray]:
    """Unit value-axis directions for a set of layers, from ``value_axis.npy``.

    ``layers`` are captured-block indices (0..n_layers-1, matching
    ``LayerActivationCapture``'s indexing) and the returned dict's keys stay in that
    space, because they are reused downstream to index the model's captured
    activations directly -- shifting them would silently read the wrong block.

    ``axis_layer_offset`` is the only place the axis file's own row-numbering
    convention enters: 0 for the original hand-built axis (no embedding row, so
    captured block i IS axis row i), 1 for the Jiang-faithful axis, which is built
    with ``output_hidden_states=True`` and therefore has an extra leading embedding
    row (65 rows for 64 blocks), making captured block i axis row i+1. Passing the
    wrong value shifts every layer by one and silently produces plausible-looking
    but wrong numbers.
    """
    axis = np.load(axis_path)
    return {
        L: unit_direction(axis[L + axis_layer_offset].astype(np.float32)) for L in layers
    }


class ActivationSink:
    """Collects per-step, per-layer mean-pooled activation vectors for probes.

    The fitted probes use the same mean-pooled activations the axis readout does,
    so they are dumped here in the single projection forward pass rather than
    replaying the model later. Vectors are stored ``float16`` to keep the ``.npz``
    manageable; metadata arrays run parallel to ``activations`` row-for-row.
    """

    def __init__(self) -> None:
        self._vecs: list[np.ndarray] = []
        self.trajectory_id: list[str] = []
        self.task_id: list[str] = []
        self.seed: list[int] = []
        self.outcome: list[int] = []
        self.step_index: list[int] = []
        self.rel_pos: list[float] = []
        self.layer: list[int] = []

    def add(
        self,
        record: TrajectoryRecord,
        step_index: int,
        rp: float,
        layer: int,
        vec: np.ndarray,
    ) -> None:
        self._vecs.append(vec.astype(np.float16))
        self.trajectory_id.append(record.trajectory_id)
        self.task_id.append(record.task_id)
        self.seed.append(-1 if record.seed is None else int(record.seed))
        self.outcome.append(int(record.outcome))
        self.step_index.append(int(step_index))
        self.rel_pos.append(float(rp))
        self.layer.append(int(layer))

    def save(self, path: Path) -> None:
        """Write activations atomically (tmp + replace) so a mid-write failure
        cannot corrupt the only on-disk checkpoint.

        Temp path must end in ``.npz`` — numpy appends ``.npz`` otherwise
        (e.g. ``foo.npz.tmp`` → ``foo.npz.tmp.npz``), which breaks replace.
        """
        import shutil

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hidden = self._vecs[0].shape[0] if self._vecs else 0
        activations = (
            np.stack(self._vecs)
            if self._vecs
            else np.zeros((0, hidden), dtype=np.float16)
        )
        # Must end with .npz so np.savez_compressed does not append another .npz
        tmp = path.with_name(path.stem + ".tmp.npz")
        stale = path.with_name(path.name + ".tmp.npz")  # leftover from older buggy path
        for p in (tmp, stale):
            if p.exists():
                p.unlink()
        try:
            np.savez_compressed(
                tmp,
                activations=activations,
                trajectory_id=np.array(self.trajectory_id),
                task_id=np.array(self.task_id),
                seed=np.array(self.seed, dtype=np.int64),
                outcome=np.array(self.outcome, dtype=np.int64),
                step_index=np.array(self.step_index, dtype=np.int64),
                rel_pos=np.array(self.rel_pos, dtype=np.float64),
                layer=np.array(self.layer, dtype=np.int64),
            )
            # Drive FUSE: os.replace can fail; shutil.move is more reliable
            if path.exists():
                path.unlink()
            shutil.move(str(tmp), str(path))
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        print(f"Wrote {activations.shape[0]} activation vectors to {path}", flush=True)

    @classmethod
    def load(cls, path: Path) -> "ActivationSink":
        """Resume from a previously written activations dump.

        Prefer local disk over a FUSE-mounted mirror: a multi-GB ``np.load`` over
        FUSE can hang for tens of minutes with no output. Prints progress so a
        notebook front-end does not look frozen.
        """
        path = Path(path)
        print(f"Loading activations from {path} ...", flush=True)
        data = np.load(path, allow_pickle=False)
        sink = cls()
        acts = np.asarray(data["activations"], dtype=np.float16)
        n = int(acts.shape[0])
        print(f"  activations shape={acts.shape}; binding metadata...", flush=True)
        # Bulk materialize — avoid a Python loop over every row for metadata.
        sink._vecs = [acts[i] for i in range(n)]
        sink.trajectory_id = [str(x) for x in np.asarray(data["trajectory_id"]).tolist()]
        sink.task_id = [str(x) for x in np.asarray(data["task_id"]).tolist()]
        sink.seed = [int(x) for x in np.asarray(data["seed"]).tolist()]
        sink.outcome = [int(x) for x in np.asarray(data["outcome"]).tolist()]
        sink.step_index = [int(x) for x in np.asarray(data["step_index"]).tolist()]
        sink.rel_pos = [float(x) for x in np.asarray(data["rel_pos"]).tolist()]
        sink.layer = [int(x) for x in np.asarray(data["layer"]).tolist()]
        print(f"  loaded {n} activation rows", flush=True)
        return sink

    def drop_trajectory(self, trajectory_id: str) -> None:
        """Remove any rows for ``trajectory_id`` (used before re-projecting)."""
        keep = [i for i, t in enumerate(self.trajectory_id) if t != trajectory_id]
        self._vecs = [self._vecs[i] for i in keep]
        self.trajectory_id = [self.trajectory_id[i] for i in keep]
        self.task_id = [self.task_id[i] for i in keep]
        self.seed = [self.seed[i] for i in keep]
        self.outcome = [self.outcome[i] for i in keep]
        self.step_index = [self.step_index[i] for i in keep]
        self.rel_pos = [self.rel_pos[i] for i in keep]
        self.layer = [self.layer[i] for i in keep]


def extract_rows_for_trajectory(
    record: TrajectoryRecord,
    model,
    tokenizer,
    *,
    layers: list[int],
    directions: dict[int, torch.Tensor],
    enable_thinking: bool,
    n_layers: int,
    device: torch.device,
    fidelity_state: dict | None = None,
    activation_sink: "ActivationSink | None" = None,
) -> list[dict]:
    """One forward pass per step; per layer emit mean/last cosine over ``G_t``.

    The step readout is the mean cosine of the value axis over every
    agent-generated token ``G_t`` (think + content + tool-call render).
    ``proj_final`` (cosine at the last generated token) is kept as a robustness
    column, and ``n_gen_tokens = |G_t|``. Steps with an empty ``G_t`` are skipped,
    but ``rel_pos``/``n_steps`` are computed over the full step list first so
    positional bins are unaffected by omissions.
    """
    rows: list[dict] = []
    n_steps = record.n_steps

    for step in record.steps:
        i = step.step_index
        rp = rel_pos(i, n_steps)

        # The native assistant turn is authoritative for token-level replay; fall
        # back to a plain-content message for legacy thinking-off trajectories.
        assistant_msg = step.assistant_message or {
            "role": "assistant",
            "content": step.assistant_response,
        }
        if step.assistant_message is None and not step.assistant_response.strip():
            continue

        messages = list(step.messages_before_gen) + [assistant_msg]
        text, input_ids_list, offset_mapping = _encode_template(
            tokenizer, messages, enable_thinking
        )
        # Verify template fidelity once on the first structured turn: the read
        # positions are only meaningful if the render reproduces generation.
        if (
            fidelity_state is not None
            and not fidelity_state.get("checked")
            and step.assistant_message is not None
        ):
            assert_render_fidelity(text, step.assistant_message)
            fidelity_state["checked"] = True

        gen_idx = generated_token_indices(text, offset_mapping)
        if not gen_idx:
            continue

        # Keep only in-range generated positions (hooks store just these tokens).
        seq_len = len(input_ids_list)
        valid = [t for t in gen_idx if t < seq_len]
        if not valid:
            continue

        input_ids = torch.tensor([input_ids_list], device=device)
        # token_indices + CPU store: avoid parking 64×seq×hidden on the GPU.
        # Backbone-only forward: skip lm_head logits (seq×vocab), which OOMs on
        # long late agent steps even when residual hooks would otherwise fit.
        capture = LayerActivationCapture(
            model,
            n_layers=n_layers,
            token_indices=valid,
            store_device="cpu",
        )
        try:
            with torch.no_grad():
                model.model(input_ids=input_ids, use_cache=False)

            for layer in layers:
                layer_act = capture.get(layer)
                if layer_act is None:
                    continue
                if layer_act.dim() == 3:
                    layer_act = layer_act[0]
                acts = layer_act.float()  # (|G_t|, hidden) already indexed
                if acts.shape[0] == 0:
                    continue
                direction = directions[layer]
                if direction.device.type != "cpu":
                    direction = direction.cpu()
                cos = cosine_projection(acts, direction)  # (|G_t|,)
                rows.append(
                    {
                        "task_id": record.task_id,
                        "trajectory_id": record.trajectory_id,
                        "seed": record.seed,
                        "outcome": record.outcome,
                        "exit_status": record.exit_status,
                        "step_index": i,
                        "n_steps": n_steps,
                        "rel_pos": rp,
                        "layer": layer,
                        "proj_mean": float(cos.mean().item()),
                        "proj_final": float(cos[-1].item()),
                        "n_gen_tokens": int(acts.shape[0]),
                    }
                )
                if activation_sink is not None:
                    activation_sink.add(
                        record, i, rp, layer, acts.mean(dim=0).numpy()
                    )
        finally:
            capture.remove()
            del input_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return rows


def _mirror_checkpoint(
    *,
    output_path: Path,
    activations_npz: Path | None,
    mirror_output: Path | None,
    mirror_activations: Path | None,
) -> None:
    """Copy local checkpoints to a durable mirror (e.g. Google Drive)."""
    import shutil

    def _copy_replace(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".mirror_tmp" + dst.suffix)
        if dst.suffix == ".npz":
            tmp = dst.with_name(dst.stem + ".mirror_tmp.npz")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(src, tmp)
        if dst.exists():
            dst.unlink()
        shutil.move(str(tmp), str(dst))

    if mirror_output is not None and output_path.exists():
        _copy_replace(output_path, mirror_output)
        print(f"    mirrored parquet -> {mirror_output}", flush=True)
    if (
        mirror_activations is not None
        and activations_npz is not None
        and activations_npz.exists()
    ):
        _copy_replace(activations_npz, mirror_activations)
        print(
            f"    mirrored npz -> {mirror_activations} "
            f"({round(mirror_activations.stat().st_size / 1e6, 1)} MB)",
            flush=True,
        )


def run(
    traj_dir: Path,
    *,
    axis_path: Path,
    layers: list[int],
    model_name: str,
    enable_thinking: bool,
    dtype: str,
    n_layers: int,
    output_path: Path,
    check_thinking: bool = True,
    activations_npz: Path | None = None,
    mirror_output: Path | None = None,
    mirror_activations: Path | None = None,
    mirror_every: int = 10,
    axis_layer_offset: int = 0,
) -> pd.DataFrame:
    records = load_trajectories_from_dir(traj_dir)
    if not records:
        raise FileNotFoundError(f"No normalized trajectories in {traj_dir}")

    if check_thinking:
        assert_thinking_mode_matches(records, enable_thinking)

    directions_np = load_axis_directions(axis_path, layers, axis_layer_offset)

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    # Cosine is run on CPU against gen-token activations (VRAM-safe for long trajs).
    directions = {
        L: torch.tensor(v, dtype=torch.float32, device="cpu")
        for L, v in directions_np.items()
    }

    activation_sink = None
    if activations_npz is not None:
        if activations_npz.exists():
            activation_sink = ActivationSink.load(activations_npz)
            print(
                f"Resumed {len(activation_sink.trajectory_id)} activation rows from "
                f"{activations_npz}",
                flush=True,
            )
        else:
            activation_sink = ActivationSink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    all_rows: list[dict] = []
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        if len(prev):
            all_rows = prev.to_dict(orient="records")
            done_ids = set(prev["trajectory_id"].astype(str).unique())
            print(
                f"Resuming: {len(done_ids)} trajectories already in {output_path}",
                flush=True,
            )

    fidelity_state: dict = {"checked": False}
    n_total = len(records)
    n_pending = sum(1 for r in records if r.trajectory_id not in done_ids)
    print(
        f"Progress: {len(done_ids)}/{n_total} already done, "
        f"{n_pending} remaining to project",
        flush=True,
    )
    if mirror_output is not None or mirror_activations is not None:
        print(
            f"Drive/durable mirror every {mirror_every} trajs "
            f"(output={mirror_output}, activations={mirror_activations})",
            flush=True,
        )
    n_skipped = 0
    for record in records:
        tid = record.trajectory_id
        n_done = len(done_ids)
        n_left = n_total - n_done
        if tid in done_ids:
            n_skipped += 1
            continue
        if n_skipped:
            print(f"  (skipped {n_skipped} already-done trajs)", flush=True)
            n_skipped = 0
        print(
            f"  [{n_done}/{n_total} done | {n_left} left] "
            f"{tid}: {record.n_steps} steps, outcome={record.outcome}",
            flush=True,
        )
        if activation_sink is not None:
            activation_sink.drop_trajectory(tid)
        rows = extract_rows_for_trajectory(
            record,
            model,
            tokenizer,
            layers=layers,
            directions=directions,
            enable_thinking=enable_thinking,
            n_layers=n_layers,
            device=device,
            fidelity_state=fidelity_state,
            activation_sink=activation_sink,
        )
        all_rows.extend(rows)
        done_ids.add(tid)
        n_done = len(done_ids)
        n_left = n_total - n_done
        print(f"    -> {len(rows)} projection rows", flush=True)

        # Checkpoint after each traj so an OOM or disconnect does not lose work.
        df = pd.DataFrame(all_rows)
        df.to_parquet(output_path, index=False)
        if activation_sink is not None:
            activation_sink.save(activations_npz)
        print(
            f"    checkpointed {len(df)} rows -> {output_path} "
            f"[{n_done}/{n_total} done | {n_left} left]",
            flush=True,
        )
        if (
            mirror_every > 0
            and (mirror_output is not None or mirror_activations is not None)
            and n_done % mirror_every == 0
        ):
            _mirror_checkpoint(
                output_path=output_path,
                activations_npz=activations_npz,
                mirror_output=mirror_output,
                mirror_activations=mirror_activations,
            )

    if n_skipped:
        print(f"  (skipped {n_skipped} already-done trajs)", flush=True)

    df = pd.DataFrame(all_rows)
    print(f"Wrote {len(df)} rows to {output_path}", flush=True)

    if activation_sink is not None:
        activation_sink.save(activations_npz)

    if mirror_output is not None or mirror_activations is not None:
        print("Final mirror to durable storage...", flush=True)
        _mirror_checkpoint(
            output_path=output_path,
            activations_npz=activations_npz,
            mirror_output=mirror_output,
            mirror_activations=mirror_activations,
        )

    return df


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traj-dir",
        type=Path,
        default=NORMALIZED_DIR,
        help="Directory of normalized trajectory JSON files",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=data_file("projections.parquet"),
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to project. Default: all layers "
        "0..n_layers-1. The primary/headline layer is a config value used by the "
        "analysis step.",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Shorthand for a single-layer sweep (equivalent to --layers <L>).",
    )
    ap.add_argument(
        "--activations-npz",
        type=Path,
        default=None,
        help="If set, also dump per-step mean-pooled activation vectors per layer "
        "to this .npz (consumed by the fitted-probe stage).",
    )
    ap.add_argument(
        "--mirror-output",
        type=Path,
        default=None,
        help="Optional durable copy path for the projections parquet (e.g. Google Drive).",
    )
    ap.add_argument(
        "--mirror-activations",
        type=Path,
        default=None,
        help="Optional durable copy path for the activations npz (e.g. Google Drive).",
    )
    ap.add_argument(
        "--mirror-every",
        type=int,
        default=10,
        help="Copy to --mirror-* every N newly completed trajs (also mirrors at end). "
        "Set 0 to disable periodic mirror (final mirror still runs if paths set).",
    )
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--axis-path", type=Path, default=defaults["axis_path"])
    ap.add_argument("--dtype", default=defaults["dtype"])
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    ap.add_argument(
        "--axis-layer-offset",
        type=int,
        default=0,
        help="Axis row = captured-block index + this offset. 0 for the original axis "
        "(no embedding row). 1 for the Jiang-faithful axis, which is embedding-inclusive "
        "(65 rows for 64 blocks), so captured block i is axis row i+1. The output 'layer' "
        "column stays in captured-block numbering either way.",
    )
    ap.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=defaults["enable_thinking"],
        help="Render the chat template with Qwen3 thinking mode. Defaults to the "
        "config value (thinking ON in Stage 2); use --no-enable-thinking to "
        "project legacy thinking-off trajectories.",
    )
    ap.add_argument(
        "--no-thinking-check",
        action="store_true",
        help="Bypass the generation/projection thinking-mode consistency check",
    )
    ap.add_argument(
        "--mock-axis",
        action="store_true",
        help="Use random unit axis (smoke test only)",
    )
    args = ap.parse_args()

    axis_path = args.axis_path
    if args.mock_axis:
        rng = np.random.default_rng(42)
        mock = rng.standard_normal((defaults["n_layers"], defaults["hidden_dim"]))
        mock = mock / np.linalg.norm(mock, axis=1, keepdims=True)
        axis_path = data_file("mock_value_axis.npy")
        np.save(axis_path, mock.astype(np.float32))
        print(f"Using mock axis at {axis_path}", flush=True)
    else:
        require_axis_path(axis_path)

    if args.layers is not None:
        layers = args.layers
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(args.n_layers))

    print(
        f"Extracting projections from {args.traj_dir} over {len(layers)} layer(s)...",
        flush=True,
    )
    run(
        args.traj_dir,
        axis_path=axis_path,
        layers=layers,
        model_name=args.model,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        n_layers=args.n_layers,
        output_path=args.output,
        check_thinking=not args.no_thinking_check,
        activations_npz=args.activations_npz,
        mirror_output=args.mirror_output,
        mirror_activations=args.mirror_activations,
        mirror_every=args.mirror_every,
        axis_layer_offset=args.axis_layer_offset,
    )


if __name__ == "__main__":
    main()
