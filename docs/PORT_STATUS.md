# Port status (INTERNAL - DELETE BEFORE RELEASE)

This file tracks the staged port from the working repository. It must not be present
in the released repository. Nothing else in this repository refers to it.

- [x] **Stage 1 - scaffold.** Layout, README, license, gitignore, pyproject, docs stubs.
- [ ] **Stage 2 - axis reconstruction.** `src/prognosis/axis/`, single-turn verification.
- [ ] **Stage 3 - agent rollouts.** `src/prognosis/agent/`, task selection, normalization,
      elicited confidence.
- [ ] **Stage 4 - extraction and projection.** `src/prognosis/extract/`.
- [ ] **Stage 5 - analysis scripts.** `src/prognosis/analysis/`, `scripts/`.
- [ ] **Stage 6 - results.** `results/` JSON and CSV the paper reads from.
- [ ] **Stage 7 - notebooks.** Highest leak surface; port last, after a full scrub.
- [ ] **Final - anonymity sweep.** See the checklist below, then delete this file.

## Anonymity checklist

Known leak sites in the source repository, to verify absent here before release:

- [ ] Author GitHub handle. Present in 25 source-repo files: 11 stage1/stage2
      notebooks, 3 single_turn_control notebooks, `docs/setup.md`,
      `docs/aws_runbook.md`, and one source file - `stage1/stage1/icrl_gen/wikipedia.py`
      sets it in a `USER_AGENT` string, which a notebook-only sweep would miss.
- [ ] Local Windows paths (`C:\Users\<name>\...`). Present in 8 source-repo files,
      including `stage1/data/axis_manifest.json` and three
      `stage2/data/normalized/*/ingest_manifest.json` provenance manifests. These
      need rewriting to relative paths, not deleting.
- [ ] `.env`. Must never be copied. `.env.example` ships instead.
- [ ] Google Drive and JupyterHub paths. Not identifying on their own, but encode a
      workflow a reviewer cannot reproduce; make them configurable roots.
- [ ] Personal correspondence. The source repo notes that the ICRL dataset was shared
      directly by an author of the prior work. The dataset is public; cite the public
      release, not the correspondence.
- [ ] Institution names, acknowledgments, funding.

Not leaks, deliberately kept: references to the prior work's public repository and
dataset. That is cited prior work the paper builds on, and removing it would break
the axis provenance.
