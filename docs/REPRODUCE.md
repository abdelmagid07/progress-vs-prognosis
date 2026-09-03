# Reproduction order

Stages 1 and 3 need a GPU. Everything else runs on CPU from the projection tables
those stages produce.

| # | Stage | GPU | Entry point |
|---|---|---|---|
| 1 | Reconstruct the value axis | yes | upstream + `patches/` |
| 1b | Single-turn coding control on DebugBench | yes | `prognosis.single_turn` |
| 2 | Generate agent trajectories, ingest, elicit confidence | no (served model) | `prognosis.agent` |
| 3 | Extract activations and project onto the axis | yes | `prognosis.extract` |
| 4 | Across-task and within-task readouts | no | `scripts/` |
| 5 | Probes, confound cascade, leakage controls | no | `scripts/` |
| 6 | Error-response and elicited-confidence analyses | no | `scripts/` |

## Paper section to script

Filled in as each stage is ported. Each row names the module that computes the
reported number and the file it writes under `data/`.

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
   than reimplemented. The result is a `(n_tokens, n_layers, 2 * n_functions)` table;
   at the sizes reported in the paper this is roughly 1.1 GB, so it is not
   distributed here. The run checkpoints every 25 conversations and resumes.

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
python -m prognosis.single_turn.prepare      # CPU: problems and rendered prompts
python -m prognosis.single_turn.run_control  # GPU: one forward pass per variant
python -m prognosis.single_turn.analyze      # CPU: AUROC, BCa CI, permutation
```

Three readouts are computed in the same pass. The window readout is the one
comparable to the original paper: it averages only over tokens after the point
where the correct and corrupted code diverge. The whole-span mean is diluted by a
positional ramp unrelated to correctness, and is reported for completeness only.


## Stage 2: agent trajectories

Rollouts run against an OpenAI-compatible endpoint serving the model, so this stage
needs a served model but no local GPU. `configs/providers/` holds the two request
shapes; select one with `MODEL_PROVIDER` rather than editing the model config.

```bash
bash scripts/run_mini_swe_batch.sh configs/selected_instances.txt
python -m prognosis.agent.ingest_batch --traj-dir data/trajectories/<run>     --format mini-swe-agent
python -m prognosis.agent.confidence
```

`ingest_batch` labels each trajectory resolved or unresolved from the run's
`results.json` and writes one normalized record per trajectory. `confidence` then
re-reads each saved prefix and elicits a verbalized P(success) in a separate forward
pass, so the trajectories under study are never modified by the elicitation.


## Stage 3: activations and projections

One forward pass per trajectory step. For every layer it records the mean cosine
between the axis and the agent-generated tokens of that step, keeps the final-token
cosine as a robustness column, and dumps the mean-pooled activation vectors the
probes are later fitted on, so the model is never replayed.

```bash
python -m prognosis.extract.project_steps --traj-dir data/normalized/<run>     --output data/projections.parquet --save-activations
```

The generated region of a step is the whole assistant block, reasoning and
tool-call render included, not just the final token. The run checkpoints per
trajectory and resumes, and can mirror both outputs to durable storage as it goes.


## Stages 4-6: analyses

Each script reads the projection tables from stage 3 and writes one JSON or CSV under
`data/`, and carries the exact command in its docstring. All are CPU-only. Grouped by
the part of the paper they support:

**Across-task readout**

| Script | Reports |
|---|---|
| `analyze_agentic_new_axis.py` | Final-step AUROC, the layer sweep, per-position bins |
| `readout_and_termination_checks.py` | Run termination and trajectory length |
| `logit_lens_replication.py` | Vocabulary ranking of the axis under a logit lens |

**Within-task readout and probes**

| Script | Reports |
|---|---|
| `paired_within_task_contrast.py` | Frozen axis within task, matched-prefix |
| `confound_cascade.py` | The cascade from pooled separation to matched pairs |
| `paired_direction_geometry.py` | Angle between the outcome direction and the axis |
| `paired_direction_structure.py` | Split-half stability of the outcome direction |
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
| `error_rule.py` | Error definition, text rules against exit status |
| `error_categories.py` | Measurable responses by error category |
| `error_event_response.py` | Trajectory-level contrast and event-aligned time course |

The probe fitting itself lives in `prognosis.analysis.fit_probes`; the shared
interval, permutation, and paired-test machinery in `prognosis.analysis.stats`.
