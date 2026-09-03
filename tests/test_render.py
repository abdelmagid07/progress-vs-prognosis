"""Offline tests for prompt rendering and code-span offset math (no real tokenizer)."""

from __future__ import annotations

from prognosis.single_turn.project import code_token_span
from prognosis.single_turn.render import make_prompt_and_full_text
from tests.fixtures.mock_problems import FakeTokenizer, char_offset_mapping

QUESTION = "Reverse a string in place."
CODE = "def reverse_string(s):\n    return s[::-1]\n"


def test_make_prompt_and_full_text_offsets_locate_the_code():
    tok = FakeTokenizer()
    full_text, start, end = make_prompt_and_full_text(
        tok, QUESTION, CODE, enable_thinking=True
    )
    assert full_text[start:end] == CODE
    assert QUESTION in full_text
    assert "```python" in full_text


def test_make_prompt_and_full_text_thinking_off_still_locates_code():
    tok = FakeTokenizer()
    full_text, start, end = make_prompt_and_full_text(
        tok, QUESTION, CODE, enable_thinking=False
    )
    assert full_text[start:end] == CODE
    assert "<think>" not in full_text


def test_code_token_span_recovers_char_offsets_under_one_char_one_token():
    tok = FakeTokenizer()
    full_text, start, end = make_prompt_and_full_text(
        tok, QUESTION, CODE, enable_thinking=True
    )
    offsets = char_offset_mapping(full_text)
    ts, te = code_token_span(offsets, start, end)
    assert ts == start
    assert te == end
    assert full_text[ts:te] == CODE


def test_code_token_span_out_of_range_returns_none_bounds():
    ts, te = code_token_span([(0, 1), (1, 2)], 10, 20)
    assert ts is None
