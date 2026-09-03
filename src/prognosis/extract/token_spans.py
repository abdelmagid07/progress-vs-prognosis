"""Token span helpers for reasoning vs tool_output projections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenSpan:
    token_index: int
    char_start: int
    char_end: int


def _char_to_token(offset_mapping: list[tuple[int, int]], char_pos: int) -> int | None:
    """Map a character position to the token index containing it."""
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos == end and end > start:
            return i
    for i in range(len(offset_mapping) - 1, -1, -1):
        start, end = offset_mapping[i]
        if end <= char_pos:
            return i
    return None


def _final_assistant_region(
    full_text: str,
    assistant_marker: str,
    turn_end_marker: str,
) -> tuple[int, int] | None:
    """Character span of the final assistant turn's *output* region.

    Starts right after the ``<|im_start|>assistant`` marker (so the role header
    itself is excluded) and ends at the following ``<|im_end|>`` (or end of
    text). Returns ``None`` if there is no assistant marker.
    """
    marker_pos = full_text.rfind(assistant_marker)
    if marker_pos < 0:
        return None
    region_start = marker_pos + len(assistant_marker)
    end_pos = full_text.find(turn_end_marker, region_start)
    region_end = end_pos if end_pos >= 0 else len(full_text)
    return region_start, region_end


def generated_token_indices(
    full_text: str,
    offset_mapping: list[tuple[int, int]],
    *,
    assistant_marker: str = "<|im_start|>assistant",
    turn_end_marker: str = "<|im_end|>",
) -> list[int]:
    """ALL token indices of the final assistant turn's generated region (``G_t``).

    The value axis is read as the *mean* cosine over every agent-generated token
    of a step, not just the last one. Those tokens are the
    whole ``<|im_start|>assistant`` … ``<|im_end|>`` block — think + post-think
    content + the tool-call render — so this returns every token index that falls
    inside that region (the role-header and trailing ``<|im_end|>`` markers are
    excluded). ``last_token_of_final_assistant`` returns the final element of this
    same set and stays available as the single-token robustness read.
    """
    region = _final_assistant_region(full_text, assistant_marker, turn_end_marker)
    if region is None:
        return []
    region_start, region_end = region
    idxs: list[int] = []
    for i, (start, end) in enumerate(offset_mapping):
        if end <= region_start:
            continue
        if start >= region_end:
            break
        idxs.append(i)
    return idxs


def last_token_of_final_assistant(
    full_text: str,
    offset_mapping: list[tuple[int, int]],
    *,
    assistant_marker: str = "<|im_start|>assistant",
    turn_end_marker: str = "<|im_end|>",
) -> TokenSpan | None:
    """Last token of the final assistant turn in a rendered chat transcript.

    Under thinking-ON the assistant turn is ``<think>...</think>`` + content +
    ``<tool_call>...</tool_call>``, so the "last token of assistant output" is
    the final token of that whole block, not of any one substring. We locate the
    last ``<|im_start|>assistant`` marker and take the last token strictly before
    the following ``<|im_end|>`` (or end of text). This is robust to reasoning
    and tool-call rendering, unlike matching a flattened response string.
    """
    region = _final_assistant_region(full_text, assistant_marker, turn_end_marker)
    if region is None:
        return None
    region_start, region_end = region

    last_tok = None
    for i, (start, end) in enumerate(offset_mapping):
        if end <= region_start:
            continue
        if start >= region_end:
            break
        last_tok = i

    if last_tok is None:
        return None

    start, end = offset_mapping[last_tok]
    return TokenSpan(token_index=last_tok, char_start=start, char_end=end)


def find_observation_message_index(
    messages: list[dict[str, str]],
    observation: str,
) -> int | None:
    """Find the message index whose content contains the observation text."""
    if not observation or not observation.strip():
        return None

    obs_stripped = observation.strip()
    obs_prefix = obs_stripped[: min(80, len(obs_stripped))]

    best_idx = None
    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].get("content", "")
        if obs_stripped in content or obs_prefix in content:
            best_idx = i
            break

    return best_idx


def last_token_of_message_content(
    full_text: str,
    message_content: str,
    offset_mapping: list[tuple[int, int]],
) -> TokenSpan | None:
    """Find last token of a specific message's content within templated full_text."""
    if not message_content:
        return None

    content = message_content.strip()
    idx = full_text.rfind(content)
    if idx < 0:
        prefix = content[: min(120, len(content))]
        idx = full_text.rfind(prefix)
        if idx < 0:
            return None

    content_end = idx + len(content)
    last_tok = None
    for i, (start, end) in enumerate(offset_mapping):
        if end <= idx:
            continue
        if start >= content_end:
            break
        last_tok = i

    if last_tok is None:
        return None

    start, end = offset_mapping[last_tok]
    return TokenSpan(token_index=last_tok, char_start=start, char_end=end)
