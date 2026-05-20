#!/usr/bin/env python3
"""
Step 2b: Classical recommender baselines (Popularity, Item-kNN CF, BPR).

Computes preference accuracy on the same held-out pairs as the LLM sweep,
inheriting eval_users / seeds / cold_start_sizes from the existing
results.json so the per-user numbers line up positionally.

Usage:
    python scripts/run_classical.py --config configs/20_05_goodreads_recsys.yaml
    python scripts/run_classical.py --config configs/20_05_goodreads_recsys.yaml --ablate-prompts
    python scripts/run_classical.py --config configs/20_05_goodreads_recsys.yaml --smoke-test

Output: outputs/<dataset>/results_classical[_<prompt>].json with the same
schema as run_sweep.py — merge into the main results.json via
scripts/merge_classical.py before plotting.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_sweep import load_jsonl, resolve_templates  # noqa: E402
from src.classical import BPR, ItemKNN, Popularity, parse_pair_text  # noqa: E402


# ── Deterministic per-user splitter ────────────────────────────────────────
#
# This mirrors scripts/run_sweep.py:UserSplitter's protocol (shuffle each
# user's pair pool with a per-(user, seed) RNG, train = first n, eval = last
# EVAL_PAIRS) but seeds the RNG via hashlib instead of Python's built-in
# hash() — the latter randomizes hash(str) and hash(tuple-containing-str) per
# process (PYTHONHASHSEED), so the LLM sweep's UserSplitter is itself non-
# reproducible across reruns. A deterministic local copy keeps classical
# results byte-identical across reruns; the protocol (same user, same n, same
# EVAL_PAIRS held out) is preserved, even though the specific held-out subset
# won't match the LLM run's per-user pairs cell-by-cell.

def _det_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary integer/string parts."""
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


class DeterministicSplitter:
    def __init__(self, dpo_by_user, eval_pairs, seed):
        self.dpo_by_user = dpo_by_user
        self.eval_pairs = eval_pairs
        self.seed = seed
        self._cache: dict = {}

    def _shuffled(self, uid):
        if uid not in self._cache:
            rng = random.Random(_det_seed("dpo", uid, self.seed))
            d = list(self.dpo_by_user[uid])
            rng.shuffle(d)
            self._cache[uid] = d
        return self._cache[uid]

    def get(self, uid, n_train):
        d = self._shuffled(uid)
        ep = self.eval_pairs
        return d[:-ep][:n_train], d[-ep:]


# ── Config + smoke handling ────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _smoke_config(config_path: str) -> str:
    p = Path(config_path)
    smoke = p.with_stem(p.stem + "_smoke")
    if not smoke.exists():
        raise FileNotFoundError(f"Smoke config not found: {smoke}")
    return str(smoke)


# ── Dataset loading + background matrix ────────────────────────────────────

def load_ratings(cfg) -> "pandas.DataFrame":
    dataset = cfg["dataset"]
    data_dir = cfg["data_dir"]
    if dataset == "20_05_goodreads_recsys":
        from src.data_goodreads import load_goodreads
        return load_goodreads(data_dir, min_reviews=cfg.get("min_reviews_per_book", 20_000))
    if dataset == "netflix":
        from src.data_netflix import load_netflix
        return load_netflix(data_dir, min_reviews=cfg.get("min_reviews_per_movie", 20_000))
    raise ValueError(f"Unknown dataset: {dataset}")


def build_background(df, eval_users: list):
    """Return (bg_matrix, title_to_idx, n_titles).

    bg_matrix is (n_bg_users, n_titles) CSR float32 of ratings. Eval users are
    excluded from the background entirely — they only show up via their n
    cold-start training pairs in the per-user fit. Items are keyed by title
    (collapsing duplicate item_ids that share a title, which is consistent with
    how pair_gen.py treats items).
    """
    eval_set = set(eval_users)
    bg = df[~df["user_id"].isin(eval_set)].copy()
    if len(bg) == 0:
        raise RuntimeError("No background ratings — every user is in eval_users.")

    titles = sorted(df["title"].unique())
    title_to_idx = {t: i for i, t in enumerate(titles)}

    bg_users = sorted(bg["user_id"].unique())
    bg_user_to_idx = {u: i for i, u in enumerate(bg_users)}

    rows = bg["user_id"].map(bg_user_to_idx).to_numpy()
    cols = bg["title"].map(title_to_idx).to_numpy()
    data = bg["rating"].to_numpy(dtype=np.float32)

    # If a (user, title) appears more than once (duplicate item_ids w/ same title),
    # take the mean rating to avoid double-counting.
    mat = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(bg_users), len(titles)),
        dtype=np.float32,
    )
    # Deduplicate (user, title) by averaging — scipy sums duplicates by default,
    # then we divide by the count.
    counts = sparse.coo_matrix(
        (np.ones_like(data, dtype=np.float32), (rows, cols)),
        shape=mat.shape,
    ).tocsr()
    summed = mat.tocsr()
    summed.data = summed.data / counts.data
    return summed, title_to_idx, len(titles)


# ── Pair-side helpers ──────────────────────────────────────────────────────

def pair_to_idx(row, title_to_idx, unknown: set) -> tuple[int, int] | None:
    """Parse a JSONL row's 'chosen' field and map both titles to indices.

    Returns None and records the missing title in `unknown` if either side
    is absent from the title->idx map. The orchestrator decides whether the
    skip rate is acceptable.
    """
    liked, disliked = parse_pair_text(row["chosen"])
    if liked not in title_to_idx or disliked not in title_to_idx:
        if liked not in title_to_idx:
            unknown.add(liked)
        if disliked not in title_to_idx:
            unknown.add(disliked)
        return None
    return title_to_idx[liked], title_to_idx[disliked]


# ── Per-prompt sweep ───────────────────────────────────────────────────────

def run_classical(cfg: dict, pair_dir: Path, out_dir: Path, source_results: dict,
                  prompt_name: str | None = None) -> dict:
    """Run all three classical methods over the eval_users/seeds/sizes inherited
    from `source_results` (a parsed results*.json from the LLM sweep)."""

    eval_users = source_results["eval_users"]
    seeds = source_results["seeds"]
    sizes = source_results["config"]["cold_start_sizes"]
    eval_pairs = source_results["config"]["eval_pairs_per_user"]

    if prompt_name:
        print(f"\n{'#' * 50}")
        print(f"  CLASSICAL — PROMPT VARIANT: {prompt_name!r}")
        print(f"{'#' * 50}")
    print(f"Eval users: {len(eval_users)}  |  Seeds: {seeds}  |  Sizes: {sizes}")

    # Build background matrix (once per dataset; identical across prompt variants
    # in principle, but cheap enough to redo per call). Note: we also clear the
    # ItemKNN sim cache so we don't accidentally reuse a stale bg_matrix's sim.
    print("\nLoading raw ratings …")
    df = load_ratings(cfg)
    bg_matrix, title_to_idx, n_titles = build_background(df, eval_users)
    ItemKNN.clear_cache()
    print(f"  bg_matrix: {bg_matrix.shape}  nnz={bg_matrix.nnz:,}  "
          f"titles={n_titles:,}")

    # Pairs (the same pair_dir the LLM sweep used for this variant)
    print(f"Loading pairs from {pair_dir} …")
    all_dpo = load_jsonl(pair_dir / "dpo_pairs.jsonl")
    dpo_by_user = defaultdict(list)
    for r in all_dpo:
        dpo_by_user[r["user_id"]].append(r)
    print(f"  DPO pairs: {len(all_dpo):,}")

    methods = ["pop", "iknn", "bpr"]
    results = {m: defaultdict(lambda: defaultdict(list)) for m in methods}

    unknown_titles: set = set()
    # Per-user eval-pair skip counts: (user, n) -> (skipped, total)
    skip_counts: dict = {}

    t0 = time.time()
    for seed in seeds:
        print(f"\n── Seed {seed} ──")
        splitter = DeterministicSplitter(dpo_by_user, eval_pairs, int(seed))

        print("  BPR pretrain …", end=" ", flush=True)
        W = BPR.pretrain(bg_matrix, seed=int(seed))
        print(f"done ({W.shape})")

        for ui, user in enumerate(eval_users):
            for n in sizes:
                tr_d, ev_d = splitter.get(user, n)
                train_pairs = [p for r in tr_d
                               if (p := pair_to_idx(r, title_to_idx, unknown_titles))
                               is not None]
                eval_idx_pairs = [p for r in ev_d
                                  if (p := pair_to_idx(r, title_to_idx, unknown_titles))
                                  is not None]

                ev_total = len(ev_d)
                ev_kept = len(eval_idx_pairs)
                skip_counts[(int(user), int(n))] = (ev_total - ev_kept, ev_total)

                # If too many eval pairs are skipped for a user, bail out — the
                # remaining accuracy is unreliable.
                if ev_total > 0 and (ev_total - ev_kept) / ev_total > 0.10:
                    raise RuntimeError(
                        f"User {user} n={n}: {ev_total - ev_kept}/{ev_total} eval "
                        f"pairs reference unknown titles (>10%). Aborting. "
                        f"Sample missing: {next(iter(unknown_titles), None)!r}"
                    )

                if not eval_idx_pairs:
                    # No usable eval pairs — record NaN to keep positional alignment
                    for m in methods:
                        results[m][n][seed].append(float("nan"))
                    continue

                user_seed = hash((int(seed), int(user), int(n))) & 0xFFFFFFFF

                pop = Popularity()
                pop.fit(bg_matrix, train_pairs, n_titles=n_titles, seed=user_seed)
                acc_pop = float(np.mean([pop.score_pair(l, d) for l, d in eval_idx_pairs]))

                iknn = ItemKNN()
                iknn.fit(bg_matrix, train_pairs, n_titles=n_titles, seed=user_seed)
                acc_iknn = float(np.mean([iknn.score_pair(l, d) for l, d in eval_idx_pairs]))

                bpr = BPR(W=W)
                bpr.fit(bg_matrix, train_pairs, n_titles=n_titles, seed=user_seed)
                acc_bpr = float(np.mean([bpr.score_pair(l, d) for l, d in eval_idx_pairs]))

                results["pop"][n][seed].append(acc_pop)
                results["iknn"][n][seed].append(acc_iknn)
                results["bpr"][n][seed].append(acc_bpr)

            print(f"  user {ui + 1}/{len(eval_users)} ({user}) "
                  f"pop={results['pop'][sizes[-1]][seed][-1]:.3f} "
                  f"iknn={results['iknn'][sizes[-1]][seed][-1]:.3f} "
                  f"bpr={results['bpr'][sizes[-1]][seed][-1]:.3f}")

    if unknown_titles:
        total_skips = sum(s for s, _ in skip_counts.values())
        total_pairs = sum(t for _, t in skip_counts.values())
        print(f"\nWARNING: {len(unknown_titles)} title(s) in pair files were missing "
              f"from raw ratings — skipped {total_skips}/{total_pairs} eval pairs.")
        for t in sorted(unknown_titles):
            print(f"  missing: {t!r}")

    elapsed = time.time() - t0

    # ── Integrity asserts ─────────────────────────────────────────────────
    for m in methods:
        for n in sizes:
            for seed in seeds:
                got = len(results[m][n][seed])
                assert got == len(eval_users), (
                    f"{m} n={n} seed={seed}: have {got} accuracies, "
                    f"expected {len(eval_users)}"
                )
    # Note: Popularity scores per item are seed-invariant, but per-user accuracy
    # is not — UserSplitter reshuffles each user's pair list per seed, so the
    # eval-pair subset itself varies across seeds (same as the LLM sweep).

    # ── Save ──────────────────────────────────────────────────────────────
    serializable = {
        m: {str(n): {str(s): list(by_seed[s]) for s in by_seed}
            for n, by_seed in results[m].items()}
        for m in results
    }
    save_data = {
        "config": cfg,
        "prompt_name": prompt_name,
        "eval_users": eval_users,
        "seeds": seeds,
        "results": serializable,
        "elapsed_minutes": elapsed / 60,
        "version": 2,
        "kind": "classical",
    }
    suffix = f"_{prompt_name}" if prompt_name else ""
    out_path = out_dir / f"results_classical{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved {out_path}  ({elapsed / 60:.2f} min)")
    return save_data


# ── CLI ────────────────────────────────────────────────────────────────────

def _find_source_results(out_dir: Path, prompt_name: str | None) -> Path:
    """Find the LLM results*.json corresponding to this prompt variant."""
    suffix = f"_{prompt_name}" if prompt_name else ""
    candidate = out_dir / f"results{suffix}.json"
    if candidate.exists():
        return candidate
    # Fall back to plain results.json if the ablation file doesn't exist
    fallback = out_dir / "results.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No source results found under {out_dir} for prompt={prompt_name!r}.\n"
        f"Looked for: {candidate}, {fallback}.\n"
        f"Run scripts/run_sweep.py first so eval_users/seeds are determined."
    )


def main():
    parser = argparse.ArgumentParser(description="Classical recommender baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--ablate-prompts", action="store_true",
        help="Run for ALL prompt templates (default: only the default one).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Swap config for its *_smoke.yaml variant.",
    )
    args = parser.parse_args()

    config_path = _smoke_config(args.config) if args.smoke_test else args.config
    if args.smoke_test:
        print(f"[smoke-test] Using config: {config_path}")

    cfg = load_config(config_path)
    pair_dir = Path(cfg["pair_dir"])
    out_dir = Path(cfg["output_dir"])

    templates = resolve_templates(cfg, args.ablate_prompts)

    for name in templates:
        # Match scripts/run_sweep.py:362-367 logic for per-variant pair_dir
        if len(templates) == 1:
            t_pair_dir = pair_dir
            prompt_name = None
        else:
            t_pair_dir = pair_dir / name
            prompt_name = name

        source_path = _find_source_results(out_dir, prompt_name)
        print(f"\nInheriting splits from: {source_path}")
        with open(source_path) as f:
            source_results = json.load(f)

        run_classical(cfg, t_pair_dir, out_dir, source_results, prompt_name=prompt_name)


if __name__ == "__main__":
    main()
