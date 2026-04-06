#!/usr/bin/env python3
"""
Step 2: Cold-start training sweep — DPO vs IPO vs SimPO vs KTO.

Usage:
    python scripts/run_sweep.py --config configs/goodreads.yaml
    python scripts/run_sweep.py --config configs/goodreads.yaml --n_eval_users 10

Results are saved as JSON for later plotting.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.device import DEVICE, TORCH_DTYPE, free_memory, reset_seeds, gpu_mb
from src.logprobs import precompute_dpo_ref, precompute_kto_ref
from src.trainers import TRAINER_REGISTRY
from src.evaluate import (
    preference_accuracy,
    compute_summary,
    print_results_table,
    run_wilcoxon_tests,
)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # CLI overrides
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ── Data splitting ────────────────────────────────────────────────────────────

class UserSplitter:
    """Deterministic, cached train/eval splits per user."""

    def __init__(self, dpo_by_user, kto_by_user, eval_pairs, seed):
        self.dpo_by_user = dpo_by_user
        self.kto_by_user = kto_by_user
        self.eval_pairs = eval_pairs
        self.seed = seed
        self._cache = {}

    def _shuffled(self, uid):
        if uid not in self._cache:
            rng_d = random.Random(hash(("dpo", uid, self.seed)))
            rng_k = random.Random(hash(("kto", uid, self.seed)))

            d = self.dpo_by_user[uid].copy()
            rng_d.shuffle(d)

            kp = defaultdict(list)
            for r in self.kto_by_user[uid]:
                kp[r["pair_id"]].append(r)
            pids = list(kp.keys())
            rng_k.shuffle(pids)
            ko = []
            for pid in pids:
                ko.extend(sorted(kp[pid], key=lambda r: (not r["label"], r["completion"])))

            self._cache[uid] = (d, ko)
        return self._cache[uid]

    def get(self, uid, n_train):
        d, k = self._shuffled(uid)
        ep = self.eval_pairs
        return (
            Dataset.from_list(d[:-ep][:n_train]),
            Dataset.from_list(d[-ep:]),
            Dataset.from_list(k[: -(ep * 2)][: n_train * 2]),
            Dataset.from_list(k[-(ep * 2) :]),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cold-start sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cold_start_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--n_eval_users", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {
        "cold_start_sizes": args.cold_start_sizes,
        "n_eval_users": args.n_eval_users,
        "num_train_epochs": args.num_train_epochs,
    })

    SIZES = cfg["cold_start_sizes"]
    N_EVAL = cfg["n_eval_users"]
    EVAL_PAIRS = cfg["eval_pairs_per_user"]
    SEED = cfg["random_seed"]
    MODEL_ID = cfg["model_id"]
    pair_dir = Path(cfg["pair_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  dtype: {TORCH_DTYPE}")
    print(f"Sizes: {SIZES}  |  Users: {N_EVAL}  |  Epochs: {cfg['num_train_epochs']}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Shared trainer kwargs
    train_kw = dict(
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        batch_size=cfg["batch_size"],
        max_grad_norm=cfg["max_grad_norm"],
        max_seq_len=cfg["max_seq_len"],
        random_seed=SEED,
    )

    # ── Load preference data ──────────────────────────────────────────────────
    print("\nLoading preference data …")
    all_dpo = load_jsonl(pair_dir / "dpo_pairs.jsonl")
    all_kto = load_jsonl(pair_dir / "kto_examples.jsonl")
    print(f"  DPO: {len(all_dpo):,}   KTO: {len(all_kto):,}")

    dpo_by_user = defaultdict(list)
    for row in all_dpo:
        dpo_by_user[row["user_id"]].append(row)

    kto_by_user = defaultdict(list)
    for row in all_kto:
        kto_by_user[row["user_id"]].append(row)

    min_dpo = max(SIZES) + EVAL_PAIRS
    eligible = [
        u for u in dpo_by_user
        if len(dpo_by_user[u]) >= min_dpo and u in kto_by_user
    ]
    assert len(eligible) >= N_EVAL, \
        f"Need {N_EVAL} users but only {len(eligible)} eligible"

    random.shuffle(eligible)
    eval_users = eligible[:N_EVAL]
    print(f"  Eval users: {eval_users}")

    splitter = UserSplitter(dpo_by_user, kto_by_user, EVAL_PAIRS, SEED)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE
        ).to(DEVICE)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    methods = ["base", "dpo", "ipo", "simpo", "kto"]
    results = {m: defaultdict(list) for m in methods}

    # Base model
    print(f"\n{'─' * 50}")
    print("Evaluating BASE model …")
    bm = load_model()
    for user in eval_users:
        _, ev, _, _ = splitter.get(user, SIZES[0])
        acc = preference_accuracy(bm, tokenizer, ev, max_length=cfg["max_seq_len"])
        for nv in SIZES:
            results["base"][nv].append(acc)
        print(f"  User {user}: {acc:.3f}")
    del bm
    free_memory()

    # Per-user training
    run_ctr = 0
    for ui, user in enumerate(eval_users):
        print(f"\n{'═' * 50}")
        print(f"  User {ui + 1}/{N_EVAL}: {user}")
        print(f"{'═' * 50}")

        for n in SIZES:
            tr_d, ev_d, tr_k, ev_k = splitter.get(user, n)
            print(f"\n  n={n}  ({len(tr_d)} DPO, {len(tr_k)} KTO rows)")

            # Reference log-probs
            print("    ref …", end=" ", flush=True)
            rm = load_model()
            ref_c, ref_r = precompute_dpo_ref(rm, tokenizer, tr_d, cfg["max_seq_len"])
            ref_k = precompute_kto_ref(rm, tokenizer, tr_k, cfg["max_seq_len"])
            del rm
            free_memory()
            print("done")

            method_specs = [
                ("dpo", dict(ref_chosen_lps=ref_c, ref_rejected_lps=ref_r,
                             beta=cfg["dpo_beta"]),
                 tr_d, ev_d, False),
                ("ipo", dict(ref_chosen_lps=ref_c, ref_rejected_lps=ref_r,
                             beta=cfg["ipo_beta"]),
                 tr_d, ev_d, False),
                ("simpo", dict(beta=cfg["simpo_beta"], gamma=cfg["simpo_gamma"]),
                 tr_d, ev_d, False),
                ("kto", dict(ref_lps=ref_k, beta=cfg["kto_beta"]),
                 tr_k, ev_k, True),
            ]

            for method, mkw, ds, ev_ds, kto_flag in method_specs:
                run_ctr += 1
                reset_seeds(SEED, run_ctr)
                print(f"    {method.upper():>5} …", end=" ", flush=True)

                m = load_model()
                TrainerCls = TRAINER_REGISTRY[method]
                t = TrainerCls(
                    **mkw, model=m, tokenizer=tokenizer,
                    dataset=ds, peft_config=lora_cfg, **train_kw,
                )
                t.train()
                acc = preference_accuracy(
                    t.model, tokenizer, ev_ds,
                    is_kto=kto_flag, max_length=cfg["max_seq_len"],
                )
                results[method][n].append(acc)
                print(f"acc={acc:.3f}  ({gpu_mb():.0f} MB)")
                del t, m
                free_memory()

            del ref_c, ref_r, ref_k

    elapsed = time.time() - t0

    # ── Save raw results ──────────────────────────────────────────────────────
    save_data = {
        "config": cfg,
        "eval_users": eval_users,
        "results": {m: dict(results[m]) for m in results},
        "elapsed_minutes": elapsed / 60,
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    mean_res, se_res = compute_summary(results, SIZES)
    print(f"\n{'═' * 50}")
    print("RESULTS SUMMARY")
    print(f"{'═' * 50}\n")
    print_results_table(mean_res, se_res, SIZES)
    run_wilcoxon_tests(results, SIZES)
    print(f"\nTotal runtime: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
