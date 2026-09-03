"""Code corruption variants for the DebugBench control.

Reimplemented from the original paper's description of the four corruption
categories ("Correlation with code correctness") — the original
`code_utils.py` this mirrors is not present in the reference snapshot
in the released reference code, so this reproduces the
same four categories, not a byte-identical replication of the paper's code.
Each function is seeded (matching the reference script's
``hashlib.md5(slug)`` convention) so a given problem always corrupts the same
way.
"""

from __future__ import annotations

import ast
import builtins
import io
import keyword
import random
import tokenize

_BUILTIN_NAMES = frozenset(dir(builtins))
_KEYWORDS = frozenset(keyword.kwlist)
_SELF_LIKE = frozenset({"self", "cls"})


def stable_seed(slug: str) -> int:
    """32-bit seed derived from a problem slug (matches the reference script)."""
    import hashlib

    return int(hashlib.md5(slug.encode()).hexdigest(), 16) % (2**31)


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def introduce_syntax_error(code: str, seed: int) -> str:
    """Remove a trailing block colon, or de-indent one line, chosen at random."""
    rng = random.Random(seed)
    lines = code.split("\n")

    colon_idx = [i for i, l in enumerate(lines) if l.rstrip().endswith(":")]
    indent_idx = [i for i, l in enumerate(lines) if _leading_ws(l)]

    pool = [("colon", i) for i in colon_idx] + [("indent", i) for i in indent_idx]
    if not pool:
        return code

    kind, i = rng.choice(pool)
    line = lines[i]
    if kind == "colon":
        lines[i] = line.rstrip()[:-1]
    else:
        ws = _leading_ws(line)
        if "\t" in ws:
            new_ws = ws.replace("\t", "", 1)
        else:
            new_ws = ws[4:] if len(ws) >= 4 else ""
        lines[i] = new_ws + line[len(ws):]
    return "\n".join(lines)


def shuffle_lines(code: str, seed: int) -> str:
    """Random permutation of the solution's lines."""
    rng = random.Random(seed)
    lines = code.split("\n")
    shuffled = lines[:]
    rng.shuffle(shuffled)
    return "\n".join(shuffled)


def _obfuscation_targets(code: str) -> set[str]:
    """Assignment/loop/parameter identifiers eligible for renaming.

    Heuristic, not full scope analysis (adequate for short, single-function
    LeetCode-style solutions): any ``Name`` node used as a store target, plus
    function parameters, excluding keywords/builtins/self/cls.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)

    return names - _SELF_LIKE - _KEYWORDS - _BUILTIN_NAMES


def obfuscate_variables(code: str, seed: int) -> str:
    """Rename local variable/parameter names to single (or short) letters."""
    targets = sorted(_obfuscation_targets(code))
    if not targets:
        return code

    rng = random.Random(seed)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    rng.shuffle(letters)
    pool = list(letters)
    i = 0
    while len(pool) < len(targets):
        pool.append(letters[i % 26] + letters[(i // 26) % 26])
        i += 1
    rename = {name: pool[idx] for idx, name in enumerate(targets)}

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return code

    new_tokens = [
        (tok.type, rename.get(tok.string, tok.string) if tok.type == tokenize.NAME else tok.string)
        for tok in tokens
    ]
    try:
        return tokenize.untokenize(new_tokens)
    except Exception:
        return code
