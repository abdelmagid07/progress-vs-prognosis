# Reproduction order

Stages 1 and 3 need a GPU. Everything else runs on CPU from the projection tables
those stages produce.

| # | Stage | GPU | Entry point |
|---|---|---|---|
| 1 | Reconstruct the value axis and verify it single-turn | yes | `notebooks/` |
| 1b | Single-turn coding control on DebugBench | yes | `prognosis.single_turn` |
| 2 | Generate agent trajectories, ingest, elicit confidence | no (served model) | `prognosis.agent` |
| 3 | Extract activations and project onto the axis | yes | `prognosis.extract` |
| 4 | Across-task and within-task readouts | no | `prognosis.analysis` |
| 5 | Probes, confound cascade, leakage controls | no | `prognosis.analysis` |
| 6 | Error-response and elicited-confidence analyses | no | `prognosis.analysis` |

## Paper section to script

Filled in as each stage is ported. Each row names the module that computes the
reported number and the file it writes under `data/`.

| Paper section | Result | Script | Output |
|---|---|---|---|
| Axis reconstruction | Held-out token-level AUROC by layer; peak layer 42, AUROC 0.836 | `python -m prognosis.axis.token_level_auroc` | `data/token_level_heldout_auroc_by_layer.csv` |
| Single-turn control | Frozen axis vs. code correctness, pooled AUROC 0.641 | `python -m prognosis.single_turn.analyze` | `data/single_turn/report/report.json` |


## Stage 1 detail: reconstructing the axis

The axis is built by the prior work's own construction code, run unmodified except
for a memory and checkpointing patch. That code is not vendored here; clone it at the
pinned commit recorded in `DATA.md` and follow its `construction/` README. The
inputs are the public ICRL dataset and Qwen3-32B; the output is a `(65, 5120)` array,
one row per hidden state including the embedding row.

Verifying the reconstruction needs the token-level held-out AUROC, which the released
construction code cannot produce -- see `DATA.md`. That statistic is computed in two
steps:

1. **Extract per-token projections** (GPU). For every classified token, record its dot
   product against each reward function's before-mean and after-mean at every layer.
   The token classification logic is the prior work's, unchanged. The result is a
   `(n_tokens, n_layers, 2 * n_functions)` table; at the sizes reported in the paper
   this is roughly 1.1 GB, so it is not distributed here.
2. **Score repeated held-out splits** (CPU). `prognosis.axis.token_level_auroc`
   consumes that table. Because the axis for a given split is a linear combination of
   per-function means, every split is evaluated from the precomputed table without
   further GPU passes.


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
