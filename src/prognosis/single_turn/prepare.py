#!/usr/bin/env python
"""CPU-only prep: DebugBench -> corrupted variants -> rendered prompts.

Writes ``problems_manifest.json``, the input to ``run_control.py`` (the GPU
step). No model weights are loaded here beyond the tokenizer (needed to
render the exact chat-template text the GPU step will re-tokenize).
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from prognosis.single_turn.config import load_defaults
from prognosis.single_turn.corrupt import (
    introduce_syntax_error,
    obfuscate_variables,
    shuffle_lines,
    stable_seed,
)
from prognosis.single_turn.data import Problem, load_debugbench
from prognosis.single_turn.paths import data_file
from prognosis.single_turn.render import make_prompt_and_full_text

CORRUPTORS = {
    "syntax_error": introduce_syntax_error,
    "shuffled": shuffle_lines,
    "obfuscated": obfuscate_variables,
}


def build_variants(solution: str, buggy_code: str, seed: int, variants: list[str]) -> dict[str, str]:
    """Corrupted code per requested variant, skipping degenerate corruptions.

    Matches the reference script's skip rule: a variant is dropped if it's
    empty or identical to the original (nothing to distinguish).
    """
    out: dict[str, str] = {}
    for name in variants:
        if name == "buggy":
            code = buggy_code.lstrip("\n")
        elif name in CORRUPTORS:
            code = CORRUPTORS[name](solution, seed)
        else:
            raise ValueError(f"Unknown corruption variant: {name}")
        if code and code != solution:
            out[name] = code
    return out


def build_manifest_from_problems(
    tokenizer,
    problems: list[Problem],
    *,
    variants: list[str],
    enable_thinking: bool,
) -> list[dict]:
    """Pure rendering step, decoupled from the dataset fetch (network-free, testable)."""
    manifest = []
    for p in problems:
        p_seed = stable_seed(p.slug)
        corrupted = build_variants(p.solution, p.buggy_code, p_seed, variants)
        if not corrupted:
            print(f"  SKIP {p.slug}: no valid corruption variants", flush=True)
            continue

        orig_text, orig_start, orig_end = make_prompt_and_full_text(
            tokenizer, p.question, p.solution, enable_thinking=enable_thinking
        )
        entry = {
            "slug": p.slug,
            "category": p.category,
            "subtype": p.subtype,
            "original": {
                "code": p.solution,
                "full_text": orig_text,
                "code_char_start": orig_start,
                "code_char_end": orig_end,
            },
            "variants": {},
        }
        for name, code in corrupted.items():
            text, start, end = make_prompt_and_full_text(
                tokenizer, p.question, code, enable_thinking=enable_thinking
            )
            entry["variants"][name] = {
                "code": code,
                "full_text": text,
                "code_char_start": start,
                "code_char_end": end,
            }
        manifest.append(entry)

    return manifest


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-name", default=defaults["dataset_name"])
    ap.add_argument("--language", default=defaults["language"])
    ap.add_argument("--n-problems", default=str(defaults["n_problems"]),
                     help="int, or 'all' for the full filtered dataset")
    ap.add_argument("--seed", type=int, default=defaults["seed"])
    ap.add_argument("--variants", nargs="+", default=defaults["variants"])
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=defaults["enable_thinking"],
        help="Render the chat template with Qwen3 thinking mode. Defaults to the "
        "config value (matches how the 32B axis was constructed).",
    )
    ap.add_argument("--output", type=Path, default=data_file("problems_manifest.json"))
    args = ap.parse_args()

    n_problems = args.n_problems if args.n_problems == "all" else int(args.n_problems)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    problems = load_debugbench(
        dataset_name=args.dataset_name, language=args.language, n_problems=n_problems,
        seed=args.seed,
    )
    print(f"Loaded {len(problems)} {args.language} problems from {args.dataset_name}", flush=True)

    manifest = build_manifest_from_problems(
        tokenizer,
        problems,
        variants=args.variants,
        enable_thinking=args.enable_thinking,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    n_variant_rows = sum(len(e["variants"]) for e in manifest)
    print(
        f"Done. {len(manifest)} problems, {n_variant_rows} corrupted variants -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
