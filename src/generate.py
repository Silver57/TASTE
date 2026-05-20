"""Greedy generation helpers for the LLM-as-judge sub-study.

All three conditions (SimPO-trained, ICL_chat, base) emit one-sentence
recommendations through the same `generate_recommendation` entry point — only
the demos vary. Decoding is greedy (temperature=0 equivalent) so reruns are
bit-identical given the same model weights.
"""

from __future__ import annotations

import torch


INSTRUCTION_TEMPLATE = (
    "In one sentence, recommend whether this reader will prefer\n"
    "  A: {title_a}\n"
    "  B: {title_b}\n"
    "and briefly justify why."
)


def render_instruction(title_a: str, title_b: str) -> str:
    return INSTRUCTION_TEMPLATE.format(title_a=title_a, title_b=title_b)


def _build_messages(demos, instruction: str) -> list[dict]:
    """Chat messages for ICL_chat (n demos) or base (no demos)."""
    messages = []
    for d in demos:
        messages.append({"role": "user", "content": d["prompt"]})
        messages.append({"role": "assistant", "content": d["chosen"]})
    messages.append({"role": "user", "content": instruction})
    return messages


@torch.inference_mode()
def generate_recommendation(
    model,
    tokenizer,
    instruction: str,
    demos: list[dict] | None = None,
    max_new_tokens: int = 80,
    max_input_tokens: int = 8192,
) -> str:
    """Greedy one-sentence generation via the model's chat template.

    Args:
        model: HF causal LM (possibly with a PEFT adapter attached).
        tokenizer: matching tokenizer; pad_token must be set.
        instruction: the final user-turn content.
        demos: optional list of {"prompt", "chosen"} dicts to prepend as
            user/assistant turns. None or [] = zero-shot.
        max_new_tokens: cap on generated tokens (one sentence ≈ 30-60).
        max_input_tokens: truncate the *input* to this length if the demo
            block overflows.

    Returns:
        The decoded continuation, with leading/trailing whitespace stripped
        and truncated at the first newline.
    """
    messages = _build_messages(demos or [], instruction)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    eos_ids = [tokenizer.eos_token_id]
    # Llama-3 instruct uses a separate "end of turn" token; include it if known.
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tokenizer.unk_token_id and eot not in eos_ids:
        eos_ids.append(eot)

    out = model.generate(
        **enc,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
    )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # One sentence: cut at the first newline if the model kept going.
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text
