"""Synthetic DebugBench-shaped fixtures for offline tests (no network, no GPU)."""

from __future__ import annotations

from prognosis.single_turn.data import Problem


class FakeTokenizer:
    """Deterministic stand-in for a HF tokenizer's chat template.

    Not Qwen's real template — offline tests only need offsets/plumbing to be
    correct, not template fidelity (that's only checked in live GPU runs, same
    split as project_steps.py's render-fidelity check).
    """

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        assert tokenize is False
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            think = "<think></think>" if enable_thinking else ""
            parts.append(f"<|assistant|>{think}")
        return "\n".join(parts) + "\n"


def char_offset_mapping(text: str) -> list[tuple[int, int]]:
    """One token per character — enough to test span math without a real tokenizer."""
    return [(i, i + 1) for i in range(len(text))]


MOCK_PROBLEMS = [
    Problem(
        slug="two-sum",
        category="logic error",
        subtype="off-by-one",
        question="Given an array of integers, return indices of the two numbers that add up to a target.",
        solution=(
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        if target - n in seen:\n"
            "            return [seen[target - n], i]\n"
            "        seen[n] = i\n"
            "    return []\n"
        ),
        buggy_code=(
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        if target - n in seen:\n"
            "            return [seen[target - n], i]\n"
            "    return []\n"  # bug: forgot to populate `seen`
        ),
    ),
    Problem(
        slug="reverse-string",
        category="syntax error",
        subtype="typo",
        question="Reverse a string in place.",
        solution=(
            "def reverse_string(s):\n"
            "    left, right = 0, len(s) - 1\n"
            "    while left < right:\n"
            "        s[left], s[right] = s[right], s[left]\n"
            "        left += 1\n"
            "        right -= 1\n"
            "    return s\n"
        ),
        buggy_code=(
            "def reverse_string(s):\n"
            "    left, right = 0, len(s) - 1\n"
            "    while left < right:\n"
            "        s[left], s[right] = s[right], s[left]\n"
            "        left += 1\n"
            "    return s\n"  # bug: `right` never decremented -> infinite loop
        ),
    ),
    Problem(
        slug="is-palindrome",
        category="logic error",
        subtype="wrong comparison",
        question="Check whether a string is a palindrome.",
        solution="def is_palindrome(s):\n    return s == s[::-1]\n",
        buggy_code="def is_palindrome(s):\n    return s == s[::1]\n",  # bug: no reverse
    ),
]
