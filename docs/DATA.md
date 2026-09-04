# Data provenance

## Working directory

The stages in `REPRODUCE.md` read and write `data/`. Running them in order builds
everything the analyses need: the reconstructed value axis, agent trajectories,
projection tables, and the derived tables the analyses read.

## Upstream construction code

The axis construction code and the ICRL dataset it is built from are third-party and
public. They are not vendored into this repository; clone the construction code at the
pinned commit below and apply the memory and checkpointing patch described in its
place in `REPRODUCE.md`. Only memory and checkpointing behavior is patched; the
token-classification logic is untouched.

- code: <https://github.com/nickjiang2378/value-axis>, commit `44ad182f1e43902858748d8d4b93835c58ddfe4c`
- data: <https://huggingface.co/datasets/nickjiang/value-axis>

The patch in `patches/` changes only how activations are accumulated and adds
checkpointing. `find_modified_text_spans`, `semantic_reward_char_spans`, and
`classify_tokens` are byte-identical to the pinned upstream file, so which tokens
count as before or after reward discovery is decided entirely by upstream code.

Of the 50 reward functions in that dataset, 48 yield usable classified tokens and
build the axis; `ends_with_exclamation` and `starts_with_question` yield none.

## Axis indexing

Two details of the axis construction are easy to get wrong and both affect which layer
a number refers to.

**The axis is embedding-inclusive.** It is built from `output_hidden_states=True`, so
row 0 is the embedding output and row `i` is the output of transformer block `i-1`.
Code that hooks `model.model.layers[i]` directly has no embedding row. Axis row `L+1`
therefore corresponds to block `L`, and the two numbering schemes must never be
compared without that shift.

**All valid reward functions build the axis.** The construction uses every valid
function in the source dataset rather than a fixed train subset. The held-out AUROC
reported for the reconstruction is a separate repeated cross-validation that does not
restrict what data builds the axis.

## Held-out AUROC metric

The released construction code computes held-out AUROC from aggregate points per
function per split, not per token, and so does not reproduce the token-level statistic
the original paper reports. A separate token-level pipeline is used for that number.
The distinction is documented where the number is reported.

## Trajectory counts

Task selection targets 500 SWE-bench Verified instances. One instance produced no
usable trajectory, so all trajectory-level analyses run on 499. Step-level analyses
run on the 12,464 steps those trajectories contain. Within-task analyses use 50
matched task pairs, each pair one resolved and one unresolved trajectory of the same
instance, truncated to the shorter arm's step count.
