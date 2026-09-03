"""Offline determinism/sanity tests for the corruption functions."""

from __future__ import annotations

from prognosis.single_turn.corrupt import (
    introduce_syntax_error,
    obfuscate_variables,
    shuffle_lines,
    stable_seed,
)

SOLUTION = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, n in enumerate(nums):\n"
    "        if target - n in seen:\n"
    "            return [seen[target - n], i]\n"
    "        seen[n] = i\n"
    "    return []\n"
)


def test_stable_seed_deterministic_and_slug_dependent():
    assert stable_seed("two-sum") == stable_seed("two-sum")
    assert stable_seed("two-sum") != stable_seed("reverse-string")


def test_introduce_syntax_error_changes_code_and_is_deterministic():
    seed = stable_seed("two-sum")
    out1 = introduce_syntax_error(SOLUTION, seed)
    out2 = introduce_syntax_error(SOLUTION, seed)
    assert out1 == out2
    assert out1 != SOLUTION
    # Either a colon got dropped or a line's indentation shrank.
    assert out1.count(":") < SOLUTION.count(":") or out1 != SOLUTION


def test_introduce_syntax_error_empty_code_is_noop():
    assert introduce_syntax_error("", 0) == ""


def test_shuffle_lines_is_a_permutation_and_deterministic():
    seed = stable_seed("two-sum")
    out1 = shuffle_lines(SOLUTION, seed)
    out2 = shuffle_lines(SOLUTION, seed)
    assert out1 == out2
    assert sorted(out1.split("\n")) == sorted(SOLUTION.split("\n"))


def test_obfuscate_variables_renames_targets_and_is_deterministic():
    seed = stable_seed("two-sum")
    out1 = obfuscate_variables(SOLUTION, seed)
    out2 = obfuscate_variables(SOLUTION, seed)
    assert out1 == out2
    assert out1 != SOLUTION
    # The function name and builtin calls must survive (untokenize's 2-tuple
    # mode doesn't preserve exact original spacing, so check tokens, not
    # substrings).
    assert "two_sum" in out1
    assert "enumerate" in out1
    # The original local variable names should no longer appear as whole tokens.
    import re

    assert "seen" not in re.findall(r"\w+", out1)


def test_obfuscate_variables_preserves_syntax_validity():
    import ast

    seed = stable_seed("two-sum")
    out = obfuscate_variables(SOLUTION, seed)
    ast.parse(out)  # raises SyntaxError if broken


def test_obfuscate_variables_no_targets_is_noop():
    code = "print('hello')\n"
    assert obfuscate_variables(code, 0) == code
