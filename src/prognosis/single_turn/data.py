"""Load and sample DebugBench (Tian et al., 2024) problems.

Public dataset on Hugging Face: ``Rtian/DebugBench``. Fields used: ``slug``,
``category``, ``subtype``, ``language``, ``question``, ``solution``,
``buggy_code`` (the reference script's ``code_quality/problems.json`` used
these same field names).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Problem:
    slug: str
    category: str
    subtype: str
    question: str
    solution: str
    buggy_code: str


def load_debugbench(
    *,
    dataset_name: str = "Rtian/DebugBench",
    language: str = "python3",
    n_problems: int | str = 150,
    seed: int = 42,
) -> list[Problem]:
    """Load, filter to one language, and (optionally) sample DebugBench problems.

    ``n_problems="all"`` keeps every filtered instance; an int takes a seeded
    random sample (without replacement) of that size, or every instance if
    fewer are available.

    The dataset's ``language`` field is a lowercase short code — ``cpp``,
    ``java``, ``python3`` (verified directly against the loaded dataset;
    *not* the title-case ``"Python"`` implied by secondary descriptions of
    the dataset card, which was wrong once already here). Matching is
    case-insensitive against that code as a defensive margin, not a promise
    it covers every possible variant.
    """
    from datasets import load_dataset

    # DebugBench's only split is "test" (there is no train/val split, despite
    # some secondary sources describing it as split-less) — confirmed by the
    # dataset builder's own error listing available splits.
    ds = load_dataset(dataset_name, split="test")

    language_lower = language.lower()
    problems: list[Problem] = []
    seen_slugs: set[str] = set()
    for row in ds:
        row_language = row.get("language")
        if not row_language or row_language.lower() != language_lower:
            continue
        slug = row.get("slug")
        solution = row.get("solution")
        buggy = row.get("buggy_code")
        if not slug or not solution or not buggy:
            continue
        # DebugBench has multiple bug categories per LeetCode problem; keep
        # the first one seen per slug so each problem contributes one pair.
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        problems.append(
            Problem(
                slug=str(slug),
                category=str(row.get("category", "")),
                subtype=str(row.get("subtype", "")),
                question=str(row.get("question", "")),
                solution=str(solution),
                buggy_code=str(buggy),
            )
        )

    problems.sort(key=lambda p: p.slug)  # deterministic order before sampling

    if n_problems == "all":
        return problems

    n = int(n_problems)
    if n >= len(problems):
        return problems

    rng = random.Random(seed)
    return rng.sample(problems, n)
