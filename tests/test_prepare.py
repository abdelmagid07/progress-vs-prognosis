"""Offline tests for manifest building (no network, no GPU)."""

from __future__ import annotations

from prognosis.single_turn.prepare import build_manifest_from_problems
from tests.fixtures.mock_problems import MOCK_PROBLEMS, FakeTokenizer

VARIANTS = ["buggy", "syntax_error", "shuffled", "obfuscated"]


def test_build_manifest_from_problems_shape():
    tok = FakeTokenizer()
    manifest = build_manifest_from_problems(
        tok, MOCK_PROBLEMS, variants=VARIANTS, enable_thinking=True
    )
    assert len(manifest) == len(MOCK_PROBLEMS)

    for entry, problem in zip(manifest, MOCK_PROBLEMS):
        assert entry["slug"] == problem.slug
        assert entry["original"]["code"] == problem.solution
        full = entry["original"]["full_text"]
        s, e = entry["original"]["code_char_start"], entry["original"]["code_char_end"]
        assert full[s:e] == problem.solution

        assert set(entry["variants"]) <= set(VARIANTS)
        assert "buggy" in entry["variants"]  # every mock problem has a distinct buggy_code
        for name, payload in entry["variants"].items():
            assert payload["code"] != problem.solution
            vf = payload["full_text"]
            vs, ve = payload["code_char_start"], payload["code_char_end"]
            assert vf[vs:ve] == payload["code"]


def test_build_manifest_is_json_serializable():
    import json

    tok = FakeTokenizer()
    manifest = build_manifest_from_problems(
        tok, MOCK_PROBLEMS, variants=VARIANTS, enable_thinking=True
    )
    json.dumps(manifest)  # raises if anything non-serializable snuck in
