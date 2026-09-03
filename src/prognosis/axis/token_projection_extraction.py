"""Per-token projections against every reward function's before/after mean.

The released construction code reduces classified tokens to per-function means and
discards the tokens themselves, so the token-level held-out statistic cannot be
recovered from its output. This runs the same loop over the same conversations and
the same token classification, but records, for each classified token and layer, its
dot product against every function's before-mean and after-mean.

The axis for a split is a linear combination of those means, so this table is enough
to score any train/held-out split of the reward functions on CPU afterwards, with no
further GPU passes. Columns run [before_f0, after_f0, before_f1, after_f1, ...] in
fn_names order. Consumed by token_level_auroc.

Requires a clone of the upstream construction code with the memory and checkpointing
patch applied, and its activation_means.pt already built; see docs/REPRODUCE.md.

Reproduce: python -m prognosis.axis.token_projection_extraction --value-axis-repo <path>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_EVERY = 25


def load_upstream(repo):
    """Import the construction code's loaders and token classifier from a clone."""
    construction = Path(repo).resolve() / "construction"
    if not construction.is_dir():
        raise FileNotFoundError(f"No construction/ directory under {repo}")
    sys.path.insert(0, str(construction))
    from extract_activations import MAX_LENGTH, classify_tokens
    from shared import (CONSTRUCTION_DATA, load_conversations, load_reward_functions,
                        load_reward_labels)
    return {
        "classify_tokens": classify_tokens,
        "max_length": MAX_LENGTH,
        "means_path": CONSTRUCTION_DATA / "activation_means.pt",
        "load_conversations": load_conversations,
        "load_reward_functions": load_reward_functions,
        "load_reward_labels": load_reward_labels,
    }


def stack_means(means, n_layers):
    """(n_layers, 2 * n_functions, hidden_dim) of function means, and their names.

    Row order within a layer is [before_f0, after_f0, before_f1, after_f1, ...].
    """
    fn_names = sorted(means)
    layers = []
    for layer in range(n_layers):
        rows = []
        for name in fn_names:
            rows.append(means[name]["before_mean"][layer].float())
            rows.append(means[name]["after_mean"][layer].float())
        layers.append(torch.stack(rows))
    return torch.stack(layers), fn_names


def checkpoint_path(out_path):
    return Path(out_path).with_suffix(".partial.npz")


def save(path, dots, reward_fns, labels, processed_ids):
    np.savez_compressed(
        path,
        dots=np.asarray(dots, dtype=np.float32),
        reward_fns=np.asarray(reward_fns),
        labels=np.asarray(labels),
        processed_ids=np.asarray(processed_ids),
    )


def extract(repo, model_name, out_path, meta_path):
    up = load_upstream(repo)
    means_path = up["means_path"]
    if not means_path.exists():
        raise FileNotFoundError(f"{means_path} not found. Run extract_activations.py first.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    print(f"Model loaded on {model.device} ({n_layers} hidden-state layers)")

    means = torch.load(means_path, map_location="cpu", weights_only=False)
    stacked, fn_names = stack_means(means, n_layers)
    print(f"Loaded means for {len(fn_names)} reward functions")

    rf_by_name = {rf["name"]: rf for rf in up["load_reward_functions"]()}
    reward_labels = up["load_reward_labels"]()
    conversations = up["load_conversations"]()
    print(f"Loaded {len(conversations)} conversations, {len(reward_labels)} labels")

    dots, reward_fns, labels, processed_ids = [], [], [], []
    ckpt = checkpoint_path(out_path)
    if ckpt.exists():
        z = np.load(ckpt)
        dots = list(z["dots"])
        reward_fns, labels = list(z["reward_fns"]), list(z["labels"])
        processed_ids = list(z["processed_ids"])
        print(f"Resuming from {ckpt}: {len(processed_ids)} conversations, {len(dots)} tokens")
    done = set(processed_ids)

    for ci, conv in enumerate(conversations):
        conv_id = conv["conversation_id"]
        if conv_id in done:
            continue
        reward_fn = conv["reward_fn"]
        conv_idx = int(conv_id.split("__conv")[1])

        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        encoding = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                             add_special_tokens=False, truncation=True,
                             max_length=up["max_length"])
        input_ids = encoding["input_ids"].to(model.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden = torch.stack(
            [outputs.hidden_states[layer][0].cpu().float() for layer in range(n_layers)])
        del outputs
        torch.cuda.empty_cache()

        token_strings = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        classifications = up["classify_tokens"](
            token_strings, offset_mapping, formatted, conv,
            reward_fn, rf_by_name[reward_fn]["type"], reward_labels, conv_idx)

        for pos, label in classifications:
            if label not in ("before", "after"):
                continue
            # (n_layers, 2 * n_functions): this token against every function mean.
            token = hidden[:, pos, :].unsqueeze(1)
            dots.append((stacked * token).sum(dim=-1).numpy())
            reward_fns.append(reward_fn)
            labels.append(label)

        processed_ids.append(conv_id)
        if len(processed_ids) % CHECKPOINT_EVERY == 0 or ci == len(conversations) - 1:
            save(ckpt, dots, reward_fns, labels, processed_ids)
            print(f"  [{len(processed_ids)}/{len(conversations)}] "
                  f"{len(dots)} tokens, checkpointed", flush=True)

    save(out_path, dots, reward_fns, labels, processed_ids)
    Path(meta_path).write_text(json.dumps({
        "fn_names": fn_names,
        "n_layers": n_layers,
        "row_order": "per layer: [before_f0, after_f0, before_f1, after_f1, ...] "
                     "matching fn_names order",
    }, indent=2), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(f"\nSaved {len(dots)} token projections to {out_path}")
    print(f"Saved column metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-token projections against function means.")
    parser.add_argument("--value-axis-repo", type=Path, required=True,
                        help="clone of the upstream construction code")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--out", type=Path, default=Path("data/token_projections.npz"))
    parser.add_argument("--meta", type=Path, default=Path("data/token_projections_meta.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extract(args.value_axis_repo, args.model, args.out, args.meta)


if __name__ == "__main__":
    main()
