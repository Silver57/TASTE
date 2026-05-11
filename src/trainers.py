"""Custom alignment trainers: DPO, IPO, SimPO, KTO, ICL."""

import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model

from .logprobs import forward_completion_logprobs


class _BaseTrainer:
    """Minimal training loop shared by all methods."""

    def __init__(self, model, tokenizer, dataset, peft_config=None,
                 learning_rate=2e-5, num_train_epochs=1,
                 batch_size=2, max_grad_norm=1.0, max_seq_len=256,
                 random_seed=42):
        self.model = get_peft_model(model, peft_config) if peft_config else model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.lr = learning_rate
        self.epochs = num_train_epochs
        self.bs = batch_size
        self.max_gn = max_grad_norm
        self.max_seq_len = max_seq_len
        self.eval_max_length = max_seq_len
        self.seed = random_seed

    def eval_dataset(self, ev_ds):
        return ev_ds

    def _compute_loss(self, batch, idx):
        raise NotImplementedError

    def train(self):
        from torch.optim import AdamW

        opt = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
        )
        self.model.train()
        indices = list(range(len(self.dataset)))

        for epoch in range(self.epochs):
            rng = random.Random(self.seed + epoch)
            rng.shuffle(indices)
            for start in range(0, len(indices), self.bs):
                idx = indices[start : start + self.bs]
                batch = self.dataset.select(idx)
                loss = self._compute_loss(batch, idx)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gn)
                opt.step()


class DPOTrainerCustom(_BaseTrainer):
    def __init__(self, ref_chosen_lps, ref_rejected_lps, beta=0.1, **kw):
        super().__init__(**kw)
        self.ref_c = ref_chosen_lps
        self.ref_r = ref_rejected_lps
        self.beta = beta

    def _compute_loss(self, batch, idx):
        P = [r["prompt"] for r in batch]
        pi_c = forward_completion_logprobs(
            self.model, self.tokenizer, P,
            [r["chosen"] for r in batch], self.max_seq_len,
        )
        pi_r = forward_completion_logprobs(
            self.model, self.tokenizer, P,
            [r["rejected"] for r in batch], self.max_seq_len,
        )
        rc = torch.tensor([self.ref_c[j] for j in idx], device=pi_c.device, dtype=pi_c.dtype)
        rr = torch.tensor([self.ref_r[j] for j in idx], device=pi_r.device, dtype=pi_r.dtype)
        return -F.logsigmoid(self.beta * ((pi_c - rc) - (pi_r - rr))).mean()


class IPOTrainerCustom(_BaseTrainer):
    def __init__(self, ref_chosen_lps, ref_rejected_lps, beta=0.1, **kw):
        super().__init__(**kw)
        self.ref_c = ref_chosen_lps
        self.ref_r = ref_rejected_lps
        self.beta = beta

    def _compute_loss(self, batch, idx):
        P = [r["prompt"] for r in batch]
        pi_c = forward_completion_logprobs(
            self.model, self.tokenizer, P,
            [r["chosen"] for r in batch], self.max_seq_len,
        )
        pi_r = forward_completion_logprobs(
            self.model, self.tokenizer, P,
            [r["rejected"] for r in batch], self.max_seq_len,
        )
        rc = torch.tensor([self.ref_c[j] for j in idx], device=pi_c.device, dtype=pi_c.dtype)
        rr = torch.tensor([self.ref_r[j] for j in idx], device=pi_r.device, dtype=pi_r.dtype)
        h = (pi_c - rc) - (pi_r - rr)
        return ((h - 1.0 / (2.0 * self.beta)) ** 2).mean()


class SimPOTrainerCustom(_BaseTrainer):
    def __init__(self, beta=2.0, gamma=0.5, **kw):
        super().__init__(**kw)
        self.beta = beta
        self.gamma = gamma

    def _compute_loss(self, batch, idx):
        P = [r["prompt"] for r in batch]
        pi_c = forward_completion_logprobs(
            self.model, self.tokenizer, P,
            [r["chosen"] for r in batch], self.max_seq_len,
        )
        pi_r = forward_completion_logprobs(
            self.model, self.tokenizer, P,
            [r["rejected"] for r in batch], self.max_seq_len,
        )
        return -F.logsigmoid(self.beta * (pi_c - pi_r) - self.gamma).mean()


class KTOTrainerCustom(_BaseTrainer):
    def __init__(self, ref_lps, beta=0.1, **kw):
        super().__init__(**kw)
        self.ref_lps = ref_lps
        self.beta = beta

    def _compute_loss(self, batch, idx):
        P = [r["prompt"] for r in batch]
        C = [r["completion"] for r in batch]
        labels = torch.tensor(
            [float(r["label"]) for r in batch], device=self.model.device
        )
        pi_lp = forward_completion_logprobs(
            self.model, self.tokenizer, P, C, self.max_seq_len,
        )
        ref_lp = torch.tensor(
            [self.ref_lps[j] for j in idx],
            device=pi_lp.device, dtype=pi_lp.dtype,
        )
        reward = pi_lp - ref_lp
        kl_ref = reward.detach().mean()
        pos_loss = -F.logsigmoid(self.beta * (reward - kl_ref))
        neg_loss = -F.logsigmoid(self.beta * (kl_ref - reward))
        return (labels * pos_loss + (1.0 - labels) * neg_loss).mean()


class ICLTrainer(_BaseTrainer):
    """No-train baseline: frozen base model conditioned on n in-prompt demos.

    Training pairs are rendered as demonstrations and prepended to each eval
    prompt. Scoring then reuses `forward_completion_logprobs` unchanged.
    """

    def __init__(self, model, tokenizer, dataset, peft_config=None,
                 icl_max_seq_len=2048, **kw):
        super().__init__(
            model=model, tokenizer=tokenizer, dataset=dataset,
            peft_config=None, **kw,
        )
        self.eval_max_length = icl_max_seq_len
        self.demos = list(dataset)

    def train(self):
        pass

    def assemble_prompt(self, eval_prompt: str) -> str:
        raise NotImplementedError

    def eval_dataset(self, ev_ds):
        rows = [
            {**r, "prompt": self.assemble_prompt(r["prompt"])}
            for r in ev_ds
        ]
        return Dataset.from_list(rows)


class ICLFlatTrainer(ICLTrainer):
    """Flat-text demos: `<prompt> <chosen>` blocks separated by blank lines."""

    def assemble_prompt(self, eval_prompt: str) -> str:
        blocks = [f"{d['prompt']} {d['chosen']}" for d in self.demos]
        blocks.append(eval_prompt)
        return "\n\n".join(blocks)


class ICLChatTrainer(ICLTrainer):
    """Chat-templated demos using the tokenizer's chat template (Llama-3)."""

    def assemble_prompt(self, eval_prompt: str) -> str:
        messages = []
        for d in self.demos:
            messages.append({"role": "user", "content": d["prompt"]})
            messages.append({"role": "assistant", "content": d["chosen"]})
        messages.append({"role": "user", "content": eval_prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# Registry for the sweep loop
TRAINER_REGISTRY = {
    "dpo": DPOTrainerCustom,
    "ipo": IPOTrainerCustom,
    "simpo": SimPOTrainerCustom,
    "kto": KTOTrainerCustom,
    "icl_flat": ICLFlatTrainer,
    "icl_chat": ICLChatTrainer,
}
