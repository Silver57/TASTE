"""Preference accuracy evaluation and statistical tests."""

from collections import defaultdict

import numpy as np

from .logprobs import batch_eval_logprobs


def preference_accuracy(model, tok, eval_dataset, is_kto=False, max_length=256):
    """Compute pairwise preference accuracy on a held-out set."""
    model.eval()

    if is_kto:
        by_pair = defaultdict(dict)
        for row in eval_dataset:
            by_pair[row["pair_id"]]["pos" if row["label"] else "neg"] = row
        valid = [v for v in by_pair.values() if "pos" in v and "neg" in v]
        if not valid:
            return 0.0
        prompts = [p["pos"]["prompt"] for p in valid]
        chosen = [p["pos"]["completion"] for p in valid]
        rejected = [p["neg"]["completion"] for p in valid]
    else:
        prompts = [r["prompt"] for r in eval_dataset]
        chosen = [r["chosen"] for r in eval_dataset]
        rejected = [r["rejected"] for r in eval_dataset]

    if not prompts:
        return 0.0

    c = batch_eval_logprobs(model, tok, prompts, chosen, max_length)
    r = batch_eval_logprobs(model, tok, prompts, rejected, max_length)
    return (c > r).sum().item() / len(prompts)


def seed_averaged(results):
    """Reduce results[method][n] cells to per-user lists.

    Accepts either the old shape (results[method][n] = list[per-user acc])
    or the new shape (results[method][n][seed] = list[per-user acc]). In the
    new shape, each user's accuracies are averaged across seeds first, keeping
    "user" as the unit of analysis.
    """
    out = {}
    for method, by_n in results.items():
        out[method] = {}
        for n, cell in by_n.items():
            if isinstance(cell, dict):
                arr = np.array([cell[s] for s in cell])
                out[method][n] = arr.mean(axis=0).tolist()
            else:
                out[method][n] = list(cell)
    return out


def compute_summary(results: dict, cold_start_sizes: list):
    """Compute mean ± SE across users (seed-averaged per user if multi-seed)."""
    normalized = seed_averaged(results)
    mean_res, se_res = {}, {}
    for method in normalized:
        ms, ss = [], []
        for nv in cold_start_sizes:
            v = np.array(normalized[method][nv])
            ms.append(v.mean())
            ss.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        mean_res[method] = ms
        se_res[method] = ss
    return mean_res, se_res


def print_results_table(mean_res, se_res, cold_start_sizes):
    """Print a formatted results table."""
    hdr = f"{'n':>6}  " + "  ".join(f"{k.upper():>12}" for k in mean_res)
    print(hdr)
    print("─" * len(hdr))
    for i, nv in enumerate(cold_start_sizes):
        parts = [f"{mean_res[k][i]:.3f}±{se_res[k][i]:.3f}" for k in mean_res]
        print(f"{nv:>6}  " + "  ".join(f"{p:>12}" for p in parts))


def run_wilcoxon_tests(results, cold_start_sizes, methods=None):
    """Pairwise Wilcoxon signed-rank tests at the largest n.

    With multi-seed results, users are paired by their seed-averaged accuracy
    so that user (not user×seed) is the unit of analysis.
    """
    from scipy.stats import wilcoxon

    normalized = seed_averaged(results)

    if methods is None:
        methods = [m for m in normalized if m != "base"]
    largest = cold_start_sizes[-1]

    n_users = len(normalized[methods[0]][largest])
    if n_users < 5:
        print(f"  (Skipping significance tests — only {n_users} users)")
        return

    print(f"\nWilcoxon tests at n={largest} ({n_users} users):")
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            a = np.array(normalized[m1][largest])
            b = np.array(normalized[m2][largest])
            d = a - b
            if np.all(d == 0):
                print(f"  {m1.upper()} vs {m2.upper()}: identical")
                continue
            _, p = wilcoxon(a, b)
            sig = " *" if p < 0.05 else ""
            print(f"  {m1.upper()} vs {m2.upper()}: Δ={d.mean():+.3f}  p={p:.4f}{sig}")
