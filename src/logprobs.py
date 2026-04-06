"""Completion-only log-probability scoring."""

import torch


def forward_completion_logprobs(model, tok, prompts, completions, max_length=512):
    """Mean log-prob over COMPLETION tokens only (prompt tokens masked out)."""
    true_prompt_lens = []
    for p in prompts:
        ids = tok(p, truncation=True, max_length=max_length,
                  add_special_tokens=True)["input_ids"]
        true_prompt_lens.append(len(ids))
    true_prompt_lens = torch.tensor(true_prompt_lens)

    full_texts = [p + " " + c for p, c in zip(prompts, completions)]
    full_enc = tok(
        full_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(model.device)

    logits = model(**full_enc).logits
    lp = torch.log_softmax(logits, dim=-1)
    token_lp = lp[:, :-1].gather(
        2, full_enc["input_ids"][:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    seq_len = full_enc["input_ids"].shape[1]
    attn_mask = full_enc["attention_mask"][:, 1:].float()
    full_lens = full_enc["attention_mask"].sum(dim=1)
    comp_start = seq_len - full_lens + true_prompt_lens.to(full_lens.device) - 1
    positions = torch.arange(seq_len - 1, device=model.device).unsqueeze(0)
    comp_mask = (positions >= comp_start.unsqueeze(1)).float()
    final_mask = comp_mask * attn_mask.to(model.device)

    return (token_lp * final_mask).sum(1) / final_mask.sum(1).clamp(min=1)


def batch_eval_logprobs(model, tok, prompts, completions,
                        max_length=512, chunk_size=4):
    """Chunked inference-mode log-prob evaluation."""
    all_scores = []
    for i in range(0, len(prompts), chunk_size):
        with torch.inference_mode():
            scores = forward_completion_logprobs(
                model, tok,
                prompts[i : i + chunk_size],
                completions[i : i + chunk_size],
                max_length,
            )
        all_scores.append(scores.cpu().float())
    return torch.cat(all_scores)


def precompute_dpo_ref(model, tok, dataset, max_length=512, chunk_size=4):
    """Pre-compute reference log-probs for DPO/IPO (chosen + rejected)."""
    model.eval()
    chosen_lps, rejected_lps = [], []
    for i in range(0, len(dataset), chunk_size):
        batch = dataset.select(range(i, min(i + chunk_size, len(dataset))))
        with torch.inference_mode():
            c = forward_completion_logprobs(
                model, tok, [r["prompt"] for r in batch],
                [r["chosen"] for r in batch], max_length,
            )
            r = forward_completion_logprobs(
                model, tok, [r["prompt"] for r in batch],
                [r["rejected"] for r in batch], max_length,
            )
        chosen_lps.extend(c.cpu().float().tolist())
        rejected_lps.extend(r.cpu().float().tolist())
    return chosen_lps, rejected_lps


def precompute_kto_ref(model, tok, dataset, max_length=512, chunk_size=4):
    """Pre-compute reference log-probs for KTO."""
    model.eval()
    ref_lps = []
    for i in range(0, len(dataset), chunk_size):
        batch = dataset.select(range(i, min(i + chunk_size, len(dataset))))
        with torch.inference_mode():
            scores = forward_completion_logprobs(
                model, tok, [r["prompt"] for r in batch],
                [r["completion"] for r in batch], max_length,
            )
        ref_lps.extend(scores.cpu().float().tolist())
    return ref_lps
