"""Render a (problem, code) pair into the exact text the model is prefilled with.

Mirrors ``code_correlation.py``'s ``make_full`` in the reference code, but
goes through ``prognosis.common.chat.apply_chat_template`` so thinking mode is
controlled the same way as everywhere else in the pipeline, and returns
character offsets for the code span instead of assuming a fixed tokenizer.
"""

from __future__ import annotations

from prognosis.common.chat import apply_chat_template

SYSTEM_PROMPT = "You are a helpful coding assistant. Write clean, correct Python code."
FENCE_OPEN = "```python\n"
FENCE_CLOSE = "\n```"


def make_prompt_and_full_text(
    tokenizer,
    question: str,
    code: str,
    *,
    enable_thinking: bool,
) -> tuple[str, int, int]:
    """Return (full_text, code_char_start, code_char_end).

    ``full_text`` is the system+user turn rendered by the chat template
    (with a generation prompt) followed by the code manually appended in a
    fenced block, as if the assistant had produced it — this is a prefill,
    not a generation. The returned offsets locate ``code`` within
    ``full_text`` in characters; ``project.py`` converts these to token
    indices after tokenizing.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Write a Python solution for the following problem:\n\n{question}\n\n"
                "Provide only the code in a ```python``` block."
            ),
        },
    ]
    prompt_text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    code_start = len(prompt_text) + len(FENCE_OPEN)
    code_end = code_start + len(code)
    full_text = f"{prompt_text}{FENCE_OPEN}{code}{FENCE_CLOSE}"
    return full_text, code_start, code_end
