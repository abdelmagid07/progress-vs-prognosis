# Reproduction order

Stages 1 and 3 need a GPU. Everything else runs on CPU from the derived files in
`results/` and the projection tables produced by stage 3.

| # | Stage | GPU | Entry point |
|---|---|---|---|
| 1 | Reconstruct the value axis and verify it single-turn | yes | `notebooks/` |
| 2 | Select tasks and generate agent trajectories | no (served model) | `scripts/` |
| 3 | Extract activations and project onto the axis | yes | `notebooks/` |
| 4 | Across-task and within-task readouts | no | `scripts/` |
| 5 | Probes, confound cascade, leakage controls | no | `scripts/` |
| 6 | Error-response and elicited-confidence analyses | no | `scripts/` |

## Paper section to script

Filled in as each stage is ported. Each row names the script that writes the file in
`results/` from which the reported number is read.

| Paper section | Result | Script | Output |
|---|---|---|---|
| | | | |
