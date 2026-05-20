#!/usr/bin/env python3
"""
Step 2: Cold-start training sweep — DPO vs IPO vs SimPO vs KTO.

Usage:
    python scripts/run_sweep.py --config configs/20_05_goodreads_recsys.yaml
    python scripts/run_sweep.py --config configs/20_05_goodreads_recsys.yaml --n_eval_users 10

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


def _smoke_config(config_path: str) -> str:
    """Derive the smoke-test config path from a regular config path."""
    p = Path(config_path)
    smoke = p.with_stem(p.stem + "_smoke")
    if not smoke.exists():
        raise FileNotFoundError(
            f"Smoke config not found: {smoke}\n"
            f"Expected a *_smoke.yaml next to {p.name}"
        )
    return str(smoke)


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

def resolve_templates(cfg, ablate: bool) -> dict[str, str]:
    """Return {name: template_string} dict — same logic as prepare_data."""
    if "prompt_templates" in cfg:
        templates = cfg["prompt_templates"]
        default = cfg.get("default_prompt", next(iter(templates)))
        if ablate:
            return templates
        return {default: templates[default]}
    return {"default": cfg["prompt_template"]}


def run_sweep(cfg, pair_dir: Path, out_dir: Path, prompt_name: str | None = None):
    """Run a single cold-start sweep. Core logic extracted from old main()."""

    SIZES = cfg["cold_start_sizes"]
    N_EVAL = cfg["n_eval_users"]
    EVAL_PAIRS = cfg["eval_pairs_per_user"]
    # Backward-compat: accept either random_seeds (list) or random_seed (scalar).
    seeds = cfg.get("random_seeds", [cfg.get("random_seed", 42)])
    SEED = seeds[0]  # used only for user-selection RNG (must be stable across seeds)
    MODEL_ID = cfg["model_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompt_name:
        print(f"\n{'#' * 50}")
        print(f"  PROMPT VARIANT: {prompt_name!r}")
        print(f"{'#' * 50}")

    print(f"Device: {DEVICE}  |  dtype: {TORCH_DTYPE}")
    print(f"Sizes: {SIZES}  |  Users: {N_EVAL}  |  Epochs: {cfg['num_train_epochs']}")
    print(f"Seeds: {seeds}")

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
    methods = ["base", "dpo", "ipo", "simpo", "kto", "icl_flat", "icl_chat"]
    # results[method][n][seed] = list[per-user accuracy]
    results = {m: defaultdict(lambda: defaultdict(list)) for m in methods}

    # Base model — frozen, deterministic forward pass. Run once on the first
    # seed's splits and replicate across all seeds.
    print(f"\n{'─' * 50}")
    print("Evaluating BASE model (once, replicated across seeds) …")
    base_splitter = UserSplitter(dpo_by_user, kto_by_user, EVAL_PAIRS, seeds[0])
    bm = load_model()
    base_user_accs = {}
    for user in eval_users:
        _, ev, _, _ = base_splitter.get(user, SIZES[0])
        acc = preference_accuracy(bm, tokenizer, ev, max_length=cfg["max_seq_len"])
        base_user_accs[user] = acc
        print(f"  User {user}: {acc:.3f}")
    del bm
    free_memory()
    for seed in seeds:
        for user in eval_users:
            for nv in SIZES:
                results["base"][nv][seed].append(base_user_accs[user])

    # Per-seed, per-user training
    for si, seed in enumerate(seeds):
        print(f"\n{'#' * 50}")
        print(f"  SEED {si + 1}/{len(seeds)}: {seed}")
        print(f"{'#' * 50}")

        splitter = UserSplitter(dpo_by_user, kto_by_user, EVAL_PAIRS, seed)
        train_kw = dict(
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["num_train_epochs"],
            batch_size=cfg["batch_size"],
            max_grad_norm=cfg["max_grad_norm"],
            max_seq_len=cfg["max_seq_len"],
            random_seed=seed,
        )

        run_ctr = 0
        for ui, user in enumerate(eval_users):
            print(f"\n{'═' * 50}")
            print(f"  Seed {seed}  |  User {ui + 1}/{N_EVAL}: {user}")
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

                icl_max_seq_len = cfg.get("icl_max_seq_len", 2048)
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
                    ("icl_flat", dict(icl_max_seq_len=icl_max_seq_len),
                     tr_d, ev_d, False),
                    ("icl_chat", dict(icl_max_seq_len=icl_max_seq_len),
                     tr_d, ev_d, False),
                ]

                for method, mkw, ds, ev_ds, kto_flag in method_specs:
                    run_ctr += 1
                    reset_seeds(seed, run_ctr)
                    print(f"    {method.upper():>5} …", end=" ", flush=True)

                    m = load_model()
                    TrainerCls = TRAINER_REGISTRY[method]
                    t = TrainerCls(
                        **mkw, model=m, tokenizer=tokenizer,
                        dataset=ds, peft_config=lora_cfg, **train_kw,
                    )
                    t.train()
                    acc = preference_accuracy(
                        t.model, tokenizer, t.eval_dataset(ev_ds),
                        is_kto=kto_flag, max_length=t.eval_max_length,
                    )
                    results[method][n][seed].append(acc)
                    print(f"acc={acc:.3f}  ({gpu_mb():.0f} MB)")
                    del t, m
                    free_memory()

                del ref_c, ref_r, ref_k

    elapsed = time.time() - t0

    # ── Save raw results ──────────────────────────────────────────────────────
    serializable_results = {
        m: {nv: dict(by_seed) for nv, by_seed in results[m].items()}
        for m in results
    }
    save_data = {
        "config": cfg,
        "prompt_name": prompt_name,
        "eval_users": eval_users,
        "seeds": seeds,
        "results": serializable_results,
        "elapsed_minutes": elapsed / 60,
        "version": 2,
    }
    suffix = f"_{prompt_name}" if prompt_name else ""
    results_path = out_dir / f"results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    mean_res, se_res = compute_summary(results, SIZES)
    print(f"\n{'═' * 50}")
    print(f"RESULTS SUMMARY{f' — {prompt_name!r}' if prompt_name else ''}")
    print(f"{'═' * 50}\n")
    print_results_table(mean_res, se_res, SIZES)
    run_wilcoxon_tests(results, SIZES)
    print(f"\nTotal runtime: {elapsed / 60:.1f} min")

    return save_data


def main():
    parser = argparse.ArgumentParser(description="Cold-start sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cold_start_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--n_eval_users", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument(
        "--ablate-prompts", action="store_true",
        help="Run sweep for ALL prompt templates (default: only the default one)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Swap config for its *_smoke.yaml variant (tiny local run)",
    )
    args = parser.parse_args()

    config_path = args.config
    if args.smoke_test:
        config_path = _smoke_config(config_path)
        print(f"[smoke-test] Using config: {config_path}")

    cfg = load_config(config_path, {
        "cold_start_sizes": args.cold_start_sizes,
        "n_eval_users": args.n_eval_users,
        "num_train_epochs": args.num_train_epochs,
    })

    pair_dir = Path(cfg["pair_dir"])
    out_dir = Path(cfg["output_dir"])

    templates = resolve_templates(cfg, args.ablate_prompts)
    all_results = {}

    for name in templates:
        # Resolve data directory for this template variant
        if len(templates) == 1:
            t_pair_dir = pair_dir
        else:
            t_pair_dir = pair_dir / name

        all_results[name] = run_sweep(
            cfg, t_pair_dir, out_dir, prompt_name=name if len(templates) > 1 else None,
        )

    if len(all_results) > 1:
        print(f"\n{'#' * 50}")
        print("  PROMPT ABLATION COMPLETE")
        print(f"{'#' * 50}")
        for name, res in all_results.items():
            print(f"  {name!r}: {res['elapsed_minutes']:.1f} min")


if __name__ == "__main__":
    main()
