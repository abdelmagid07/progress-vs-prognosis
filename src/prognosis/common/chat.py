"""Qwen3 chat template helpers."""

from transformers import PreTrainedTokenizer


def apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def tokenize_chat(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
):
    text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
