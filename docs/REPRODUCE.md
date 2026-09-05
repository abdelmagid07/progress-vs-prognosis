# Reproduction order

Stages 1 and 3 need a GPU. Everything else runs on CPU from the projection tables
those stages produce.

| # | Stage | GPU | Entry point |
|---|---|---|---|
| 1 | Reconstruct the value axis | yes | upstream + `patches/` |
| 1b | Single-turn coding control on DebugBench | yes | `prognosis.single_turn` |
| 2 | Generate agent trajectories, ingest, elicit confidence | no (served model) | `prognosis.agent` |
| 2b | Build the within-task pairs | no (served model) | `scripts/` |
| 3 | Extract activations and project onto the axis | yes | `prognosis.extract` |
| 4 | Across-task and within-task readouts | no | `scripts/` |
| 5 | Probes, confound cascade, leakage controls | no | `scripts/` |
| 6 | Error-response and elicited-confidence analyses | no | `scripts/` |

## Paper section to script

Each row names the module that computes the reported number and the file it
writes under `data/`. The stage 4-6 analyses are tabulated in full below.

| Paper section | Result | Script | Output |
|---|---|---|---|
| Axis reconstruction | Held-out token-level AUROC by layer; peak layer 42, AUROC 0.836 | `python -m prognosis.axis.token_level_auroc` | `data/token_level_heldout_auroc_by_layer.csv` |
| Single-turn control | Frozen axis vs. code correctness, pooled AUROC 0.641 | `python -m prognosis.single_turn.analyze` | `data/single_turn/report/report.json` |


## Stage 1 detail: reconstructing the axis

The axis is built by the prior work's own construction code. That code is not
vendored here; clone it at the pinned commit recorded in `DATA.md`, apply the patch
below, and follow its `construction/` README. The inputs are the public ICRL dataset
and Qwen3-32B; the output is a `(65, 5120)` array, one row per hidden state including
the embedding row.

```bash
git clone https://github.com/nickjiang2378/value-axis
git -C value-axis checkout 44ad182f1e43902858748d8d4b93835c58ddfe4c
patch -p1 -d value-axis < patches/extract_activations_memory_checkpoint.patch
cd value-axis/construction
python extract_activations.py --model Qwen/Qwen3-32B
python compute_vector.py
```

The patch exists because the released script holds every classified token's per-layer
activation in a growing list for the whole run and reduces to a mean only at the end,
which exhausts memory partway through a 32B run. It accumulates a running sum and
count instead, which is the same mean in bounded space, and checkpoints periodically
so an interrupted run resumes. The token-classification functions are untouched;
see `DATA.md`.

Verifying the reconstruction needs the token-level held-out AUROC, which the released
construction code cannot produce -- see `DATA.md`. That statistic is computed in two
steps:

1. **Extract per-token projections** (GPU). For every classified token, record its dot
   product against each reward function's before-mean and after-mean at every layer.
   The token classification is the upstream function, imported from the clone rather
   than reimplemented. The result is a `(n_tokens, n_layers, 2 * n_functions)` table,
   roughly 1.1 GB at the sizes reported in the paper, so allow disk for it. The run
   checkpoints every 25 conversations and resumes.

   ```bash
   python -m prognosis.axis.token_projection_extraction --value-axis-repo value-axis
   ```

2. **Score repeated held-out splits** (CPU). Because the axis for a given split is a
   linear combination of per-function means, every split is evaluated from that table
   without further GPU passes.

   ```bash
   python -m prognosis.axis.token_level_auroc        --projections data/token_projections.npz        --meta data/token_projections_meta.json        --out data/token_level_heldout_auroc_by_layer.csv
   ```


## Stage 1b: single-turn coding control

Checks that the frozen axis still behaves as expected on single-turn code before
any agentic transfer is attempted. Each DebugBench problem contributes its correct
solution and four corrupted variants; the axis should score the correct one higher.

```bash
python -m prognosis.single_turn.prepare                            # CPU: problems and rendered prompts
python -m prognosis.single_turn.run_control --axis-layer-offset 1  # GPU: one forward pass per variant
python -m prognosis.single_turn.analyze                            # CPU: AUROC, BCa CI, permutation
```

``--axis-layer-offset 1`` is required here for the same reason as in stage 3: the
axis is embedding-inclusive, so captured block ``i`` is axis row ``i+1``. The
reported primary layer 36 is a captured block, i.e. axis row 37.

Three readouts are computed in the same pass. The window readout is the one
comparable to the original paper: it averages only over tokens after the point
where the correct and corrupted code diverge. The whole-span mean is diluted by a
positional ramp unrelated to correctness, and is reported for completeness only.


## Stage 2: agent trajectories

Rollouts run against an OpenAI-compatible endpoint serving the model, so this stage
needs a served model but no local GPU. `configs/providers/` holds the two request
shapes; select one with `MODEL_PROVIDER` rather than editing the model config.

```bash
bash scripts/run_mini_swe_batch.sh
python -m prognosis.agent.ingest_batch --traj-dir data/trajectories/<run>     --format mini-swe-agent
python -m prognosis.agent.confidence
```

`ingest_batch` labels each trajectory resolved or unresolved from the run's
`results.json` and writes one normalized record per trajectory. `confidence` then
re-reads each saved prefix and elicits a verbalized P(success) in a separate forward
pass, so the trajectories under study are never modified by the elicitation.


## Stage 2b: within-task pairs

Half the analyses compare two trajectories of the *same* task, one resolved and one not. The
main run supplies the resolved arm for every task it solved; the failing arm has to be
generated by resampling the same task with a fresh seed.

```bash
python scripts/select_pair_tasks.py       # tasks that need a failing arm -> data/pair_instances.txt
python scripts/generate_pair_arms.py --dry-run
python scripts/generate_pair_arms.py --workers 8
python scripts/package_pairs.py --main-run-normalized data/normalized/<run>
```

`generate_pair_arms.py` is a driver rather than one batch call, because whether an attempt is
usable is only known after the SWE-bench harness evaluates it: the loop is generate, evaluate,
keep the failures, retry the rest. It is resumable and caps attempts per task. Use the same
model provider that produced the main run, or the two arms of a pair differ by more than
sampling and the contrast is no longer clean.

Project the packaged pairs exactly as in stage 3, then put the table into the layer numbering
the paired analyses expect:

```bash
python scripts/normalize_pair_layers.py
python -m prognosis.agent.confidence --traj-dir data/normalized_pairs     --output data/confidence_pairs50.parquet
```

That yields the four paired inputs the analyses default to: `projections_pairs50.parquet`,
`pairs50_meanpool.npz`, `normalized_pairs.zip`, and `confidence_pairs50.parquet`.

## Stage 3: activations and projections

One forward pass per trajectory step. For every layer it records the mean cosine
between the axis and the agent-generated tokens of that step, keeps the final-token
cosine as a robustness column, and dumps the mean-pooled activation vectors the
probes are later fitted on, so the model is never replayed.

```bash
python -m prognosis.extract.project_steps --traj-dir data/normalized/<run>     --output data/projections_full.parquet --activations-npz data/full_meanpool.npz     --axis-layer-offset 1
python scripts/normalize_pair_layers.py --path data/projections_full.parquet
```

``--axis-layer-offset 1`` is required, not optional. The reconstructed axis is
embedding-inclusive (65 rows for 64 blocks), so captured block ``i`` is axis row
``i+1``; the flag defaults to ``0``, which is correct only for an axis with no
embedding row. Omitting it projects onto the wrong axis row at every layer while
still writing a table the layer guard accepts, because the guard checks the
``layer`` column's numbering, not which axis row the direction came from.

The generated region of a step is the whole assistant block, reasoning and
tool-call render included, not just the final token. The run checkpoints per
trajectory and resumes, and can mirror both outputs to durable storage as it goes.

The normalisation step is required, not optional. ``project_steps`` writes ``layer``
in captured-block numbering (0..63); every analysis below reads ``layer`` as the
axis row (1..64). Skipping it shifts every layer by one and produces plausible
numbers at the wrong depth.


## Stages 4-6: analyses

Each script reads the projection tables from stage 3 and writes one JSON or CSV under
`data/`, and carries the exact command in its docstring. All are CPU-only, and all run
on their defaults with no arguments once stage 3 has produced its outputs -- except
`logit_lens_replication.py`, which additionally needs `--shard` pointing at the model
`.safetensors` shard that holds `lm_head.weight`, since that lives in your Hugging Face
cache rather than under `data/`.

Two of them write CSVs that others consume, so run these first:

```bash
python scripts/paired_opening_move_features.py   # -> data/paired_opening_move_features.csv
python scripts/paired_probe_ceiling.py           # -> data/paired_probe_ceiling.csv
```

The length-controlled and observable-controlled probes read the first; the confound
cascade reads the second. The rest are independent and can run in any order.

Grouped by the part of the paper they support:

**Across-task readout**

| Script | Reports |
|---|---|
| `analyze_agentic.py` | Final-step AUROC, the layer sweep, per-position bins |
| `readout_and_termination_checks.py` | Run termination and trajectory length |
| `logit_lens_replication.py` | Vocabulary ranking of the axis under a logit lens |

**Within-task readout and probes**

| Script | Reports |
|---|---|
| `paired_within_task_contrast.py` | Frozen axis within task, matched-prefix |
| `confound_cascade.py` | The cascade from pooled separation to matched pairs |
| `paired_direction_geometry.py` | Split-half stability, and the angle to the value axis |
| `paired_probe_ceiling.py` | Within-task probe grid; feeds the cascade |
| `axis_vs_text_incremental.py` | Combined axis-plus-text model |

**Leakage controls**

| Script | Reports |
|---|---|
| `text_only_baseline.py` | Trajectory-level text baseline |
| `paired_opening_move_features.py` | Step-0 text baseline |
| `paired_length_controlled_probe.py` | Length residualization |
| `paired_observable_controlled_probe.py` | Nine-feature observable residualization |
| `paired_axis_residualised_probe.py` | Residualizing against the value axis |

**Error response**

| Script | Reports |
|---|---|
| `error_rule.py` | Shared error definition imported by the two scripts below (no output of its own) |
| `error_categories.py` | Measurable responses by error category |
| `error_event_response.py` | Trajectory-level contrast and event-aligned time course |
| `error_response_outcome_direction.py` | Both directions on both readouts: the double dissociation |

The probe fitting itself lives in `prognosis.analysis.fit_probes`; the shared
interval, permutation, and paired-test machinery in `prognosis.analysis.stats`.
The full-benchmark probe grid (the decodability reference the paired analyses
quote) is produced by:

```bash
python -m prognosis.analysis.fit_probes --activations data/full_meanpool.npz
```


## Inputs and where each comes from

Every input the analyses default to is produced by a stage above:

| Input | Produced by |
|---|---|
| `value_axis.npy` | stage 1, upstream construction code plus `patches/` |
| `token_projections.npz` | `prognosis.axis.token_projection_extraction` |
| `normalized/` | `prognosis.agent.ingest_batch` |
| `projections_full.parquet` | `prognosis.extract.project_steps` |
| `full_meanpool.npz` | `project_steps --activations-npz` |
| `confidence.parquet` | `prognosis.agent.confidence` |
| `pair_instances.txt` | `scripts/select_pair_tasks.py` |
| `normalized_pairs.zip` | `scripts/package_pairs.py` |
| `projections_pairs50.parquet`, `pairs50_meanpool.npz` | `project_steps` on the packaged pairs, then `scripts/normalize_pair_layers.py` |
| `confidence_pairs50.parquet` | `prognosis.agent.confidence` on the packaged pairs |
| `paired_opening_move_features.csv` | `scripts/paired_opening_move_features.py` |
| `paired_probe_ceiling.csv` | `scripts/paired_probe_ceiling.py` |
