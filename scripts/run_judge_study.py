#!/usr/bin/env python3
"""LLM-as-judge sub-study: SimPO@n=100 vs ICL_chat@n=10 vs base Llama-3.

Stage 1 — Generation:
    Pick 10 GoodReads users (first 10 of the main sweep's eval_users) and 5
    held-out pairs each. For every (user, pair) cell, produce a one-sentence
    recommendation under three conditions and append it to disk.

Stage 2 — Judging:
    For each cell, build three pairwise matchups with randomised A/B order and
    have Claude Sonnet 4.6 (temperature=0) return {"choice", "reason"}.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/run_judge_study.py --config configs/goodreads_judge.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.device import DEVICE, TORCH_DTYPE, free_memory, reset_seeds
from src.generate import render_instruction, generate_recommendation
from src.trainers import SimPOTrainerCustom
from src.judge import Judge
from run_sweep import UserSplitter, load_jsonl  # type: ignore


CONDITIONS = ["simpo", "icl_chat", "base"]
MATCHUPS = [("simpo", "icl_chat"), ("simpo", "base"), ("icl_chat", "base")]

_PAIR_RE = re.compile(r"^Approved:\s*(.*?),\s*Denied:\s*(.*)$")


def parse_approved(text: str) -> tuple[str, str]:
    """Return (liked_title, disliked_title) from a 'chosen' field."""
    m = _PAIR_RE.match(text.strip())
    if not m:
        raise ValueError(f"Unparseable chosen text: {text!r}")
    return m.group(1).strip(), m.group(2).strip()


def ab_titles(liked: str, disliked: str, pair_id: str) -> tuple[str, str]:
    """Deterministic per-pair A/B ordering for the generation instruction.

    Independent of the original prompt's coin flip; lets us avoid fragile
    prompt re-parsing. Stable across conditions for the same pair_id.
    """
    rng = random.Random(hash(("ab_order", pair_id)))
    return (liked, disliked) if rng.random() < 0.5 else (disliked, liked)


def build_profile(train_rows: list[dict], k: int, seed: int) -> tuple[list[str], list[str]]:
    """Pick k liked + k disliked titles from the user's training pairs.

    Uses set semantics (de-duped) and a deterministic shuffle so the same user
    yields the same profile across runs.
    """
    liked, disliked = set(), set()
    for r in train_rows:
        l, d = parse_approved(r["chosen"])
        liked.add(l)
        disliked.add(d)
    rng = random.Random(hash(("profile", seed)))
    liked_l, disliked_l = sorted(liked), sorted(disliked)
    rng.shuffle(liked_l)
    rng.shuffle(disliked_l)
    return liked_l[:k], disliked_l[:k]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_main_eval_users(main_results_path: Path) -> list:
    with open(main_results_path) as f:
        data = json.load(f)
    users = data.get("eval_users")
    if not users:
        raise RuntimeError(f"No eval_users in {main_results_path}")
    return users


def build_model(model_id: str):
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=TORCH_DTYPE
    ).to(DEVICE)


# ── Stage 1: generation ─────────────────────────────────────────────────────


def stage_generate(cfg: dict, out_dir: Path) -> Path:
    """Generate recommendations for all (user, pair, condition). Returns path
    to the JSONL written."""

    pair_dir = Path(cfg["pair_dir"])
    seed = cfg["random_seed"]
    all_dpo = load_jsonl(pair_dir / "dpo_pairs.jsonl")
    all_kto = load_jsonl(pair_dir / "kto_examples.jsonl")
    print(f"  Loaded {len(all_dpo):,} DPO rows, {len(all_kto):,} KTO rows")

    dpo_by_user: dict = defaultdict(list)
    for row in all_dpo:
        dpo_by_user[row["user_id"]].append(row)
    kto_by_user: dict = defaultdict(list)
    for row in all_kto:
        kto_by_user[row["user_id"]].append(row)

    # Reuse the main sweep's user selection.
    main_users = load_main_eval_users(Path(cfg["main_results_path"]))
    users = list(main_users)[: cfg["n_judge_users"]]
    print(f"  Users: {users}")

    splitter = UserSplitter(
        dpo_by_user, kto_by_user, cfg["eval_pairs_per_user"], seed,
    )

    n_train = cfg["simpo_n_train"]
    n_demos = cfg["icl_n_demos"]
    n_pairs = cfg["n_judge_pairs"]
    profile_k = cfg["profile_size"]
    max_new = cfg["gen_max_new_tokens"]
    icl_max = cfg["icl_max_seq_len"]

    # Per-user pre-computed slices.
    user_data: dict = {}
    for u in users:
        # Use the largest training slice we need (=n_train) so eval slice is
        # consistent with what the main sweep saw at n=100.
        tr, ev, _, _ = splitter.get(u, n_train)
        tr_rows = list(tr)
        ev_rows = list(ev)[:n_pairs]
        # Extract per-pair ground truth + stable A/B titles.
        pairs = []
        for r in ev_rows:
            gt_liked, gt_disliked = parse_approved(r["chosen"])
            ta, tb = ab_titles(gt_liked, gt_disliked, r["pair_id"])
            pairs.append({
                "pair_id": r["pair_id"],
                "gt_liked": gt_liked,
                "gt_disliked": gt_disliked,
                "title_a": ta,
                "title_b": tb,
                "instruction": render_instruction(ta, tb),
            })
        liked_prof, disliked_prof = build_profile(tr_rows, profile_k, seed=u)
        user_data[u] = {
            "train_rows": tr_rows,
            "pairs": pairs,
            "icl_demos": tr_rows[:n_demos],
            "profile_liked": liked_prof,
            "profile_disliked": disliked_prof,
        }

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    gen_path = out_dir / "judge_generations.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fresh file each run — no cache resume.
    if gen_path.exists():
        gen_path.unlink()

    def append(row: dict):
        with open(gen_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ── Phase 1: base + ICL_chat (single base model shared across users) ─────
    print(f"\n[Phase 1] base + icl_chat | device={DEVICE} dtype={TORCH_DTYPE}")
    bm = build_model(cfg["model_id"])
    bm.eval()
    for u in users:
        ud = user_data[u]
        for p in ud["pairs"]:
            rec = generate_recommendation(
                bm, tokenizer, p["instruction"],
                demos=None,
                max_new_tokens=max_new,
                max_input_tokens=icl_max,
            )
            append({
                "user_id": u, "pair_id": p["pair_id"],
                "condition": "base",
                "title_a": p["title_a"], "title_b": p["title_b"],
                "gt_liked": p["gt_liked"], "gt_disliked": p["gt_disliked"],
                "recommendation": rec,
            })
            rec_icl = generate_recommendation(
                bm, tokenizer, p["instruction"],
                demos=ud["icl_demos"],
                max_new_tokens=max_new,
                max_input_tokens=icl_max,
            )
            append({
                "user_id": u, "pair_id": p["pair_id"],
                "condition": "icl_chat",
                "title_a": p["title_a"], "title_b": p["title_b"],
                "gt_liked": p["gt_liked"], "gt_disliked": p["gt_disliked"],
                "recommendation": rec_icl,
            })
        print(f"  user {u}: base + icl_chat done ({len(ud['pairs'])} pairs)")
    del bm
    free_memory()

    # ── Phase 2: SimPO per user ──────────────────────────────────────────────
    print("\n[Phase 2] simpo (fresh adapter per user)")
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    train_kw = dict(
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        batch_size=cfg["batch_size"],
        max_grad_norm=cfg["max_grad_norm"],
        max_seq_len=cfg["max_seq_len"],
        random_seed=seed,
    )

    from datasets import Dataset

    for ui, u in enumerate(users):
        ud = user_data[u]
        print(f"  user {u} ({ui + 1}/{len(users)}) — training SimPO@n={n_train}")
        reset_seeds(seed, ui + 1)
        m = build_model(cfg["model_id"])
        tr_ds = Dataset.from_list(ud["train_rows"][:n_train])
        trainer = SimPOTrainerCustom(
            beta=cfg["simpo_beta"], gamma=cfg["simpo_gamma"],
            model=m, tokenizer=tokenizer,
            dataset=tr_ds, peft_config=lora_cfg, **train_kw,
        )
        trainer.train()
        trainer.model.eval()
        for p in ud["pairs"]:
            rec = generate_recommendation(
                trainer.model, tokenizer, p["instruction"],
                demos=None,
                max_new_tokens=max_new,
                max_input_tokens=icl_max,
            )
            append({
                "user_id": u, "pair_id": p["pair_id"],
                "condition": "simpo",
                "title_a": p["title_a"], "title_b": p["title_b"],
                "gt_liked": p["gt_liked"], "gt_disliked": p["gt_disliked"],
                "recommendation": rec,
            })
        del trainer, m
        free_memory()

    # ── Persist per-user profile sidecar (used by the judge) ─────────────────
    profile_path = out_dir / "judge_profiles.json"
    with open(profile_path, "w") as f:
        json.dump(
            {
                str(u): {
                    "liked": user_data[u]["profile_liked"],
                    "disliked": user_data[u]["profile_disliked"],
                }
                for u in users
            },
            f, indent=2,
        )
    print(f"\nGenerations → {gen_path}")
    print(f"Profiles    → {profile_path}")
    return gen_path


# ── Stage 2: judging ────────────────────────────────────────────────────────


def stage_judge(cfg: dict, out_dir: Path) -> Path:
    gen_path = out_dir / "judge_generations.jsonl"
    profile_path = out_dir / "judge_profiles.json"
    with open(profile_path) as f:
        profiles = json.load(f)

    gens = [json.loads(line) for line in open(gen_path)]
    by_cell: dict = defaultdict(dict)
    for g in gens:
        by_cell[(g["user_id"], g["pair_id"])][g["condition"]] = g

    judge = Judge(model=cfg["judge_model"], max_tokens=cfg["judge_max_tokens"])
    order_rng = random.Random(cfg["judge_order_seed"])

    out_path = out_dir / "judge_results.jsonl"
    if out_path.exists():
        out_path.unlink()

    def append(row: dict):
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    n_total = len(by_cell) * len(MATCHUPS)
    n_done = 0
    print(f"\n[Stage 2] judging {n_total} matchups via {cfg['judge_model']}")

    for (user_id, pair_id), recs in by_cell.items():
        prof = profiles[str(user_id)]
        meta = recs[CONDITIONS[0]]  # any condition has the same metadata
        for cond_x, cond_y in MATCHUPS:
            # Randomise A/B assignment between conditions, deterministically.
            if order_rng.random() < 0.5:
                cond_a, cond_b = cond_x, cond_y
            else:
                cond_a, cond_b = cond_y, cond_x

            t0 = time.time()
            try:
                resp = judge.judge(
                    liked=prof["liked"],
                    disliked=prof["disliked"],
                    gt_liked=meta["gt_liked"],
                    gt_disliked=meta["gt_disliked"],
                    rec_a=recs[cond_a]["recommendation"],
                    rec_b=recs[cond_b]["recommendation"],
                )
                err = None
                choice = resp.choice
                reason = resp.reason
                raw = resp.raw
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                choice, reason, raw = "", "", ""
                print(f"  ! judge error on ({user_id}, {pair_id}, {cond_x}vs{cond_y}): {err}")

            winner = recs[cond_a]["condition"] if choice == "A" else (
                recs[cond_b]["condition"] if choice == "B" else None
            )
            append({
                "user_id": user_id,
                "pair_id": pair_id,
                "matchup": f"{cond_x}_vs_{cond_y}",
                "cond_a": cond_a,
                "cond_b": cond_b,
                "rec_a": recs[cond_a]["recommendation"],
                "rec_b": recs[cond_b]["recommendation"],
                "gt_liked": meta["gt_liked"],
                "gt_disliked": meta["gt_disliked"],
                "choice": choice,
                "winner": winner,
                "reason": reason,
                "raw": raw,
                "error": err,
                "elapsed_s": round(time.time() - t0, 2),
            })
            n_done += 1
            if n_done % 10 == 0 or n_done == n_total:
                print(f"  {n_done}/{n_total} judged")

    print(f"Judgments → {out_path}")
    return out_path


# ── Stage 3: summary ────────────────────────────────────────────────────────


def stage_summary(out_dir: Path) -> None:
    results = [json.loads(line) for line in open(out_dir / "judge_results.jsonl")]

    per_matchup: dict = defaultdict(lambda: defaultdict(int))
    wins_per_condition: dict = defaultdict(int)
    games_per_condition: dict = defaultdict(int)
    errors = 0

    for r in results:
        if r["error"]:
            errors += 1
            continue
        m = r["matchup"]
        cond_x, cond_y = m.split("_vs_")
        per_matchup[m]["games"] += 1
        if r["winner"] == cond_x:
            per_matchup[m][f"wins_{cond_x}"] += 1
        elif r["winner"] == cond_y:
            per_matchup[m][f"wins_{cond_y}"] += 1
        wins_per_condition[r["winner"]] += 1
        games_per_condition[cond_x] += 1
        games_per_condition[cond_y] += 1

    summary = {
        "n_judgments": len(results),
        "n_errors": errors,
        "per_matchup": {m: dict(v) for m, v in per_matchup.items()},
        "win_rate_overall": {
            c: (wins_per_condition[c] / games_per_condition[c])
            if games_per_condition[c] else None
            for c in CONDITIONS
        },
    }
    path = out_dir / "judge_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Summary ──")
    print(f"  Judgments: {summary['n_judgments']}  (errors: {errors})")
    print("  Per-matchup wins:")
    for m, v in summary["per_matchup"].items():
        print(f"    {m}: {dict(v)}")
    print("  Overall win-rate (wins / games across matchups it appeared in):")
    for c, wr in summary["win_rate_overall"].items():
        wr_s = f"{wr:.3f}" if wr is not None else "n/a"
        print(f"    {c:>8}: {wr_s}")
    print(f"\nSummary → {path}")


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge sub-study")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])

    print(f"== Stage 1: generation → {out_dir} ==")
    stage_generate(cfg, out_dir)

    print(f"\n== Stage 2: judging ==")
    stage_judge(cfg, out_dir)

    print(f"\n== Stage 3: summary ==")
    stage_summary(out_dir)


if __name__ == "__main__":
    main()
