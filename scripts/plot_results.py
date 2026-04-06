#!/usr/bin/env python3
"""
Step 3: Generate figures from saved results.

Usage:
    python scripts/plot_results.py --results outputs/goodreads/results.json
    python scripts/plot_results.py --results outputs/goodreads/results.json --out figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import compute_summary


STYLES = {
    "base":  dict(color="gray",       linestyle="--", marker=""),
    "dpo":   dict(color="steelblue",  linestyle="-",  marker="o"),
    "ipo":   dict(color="darkorange", linestyle="-",  marker="s"),
    "simpo": dict(color="seagreen",   linestyle="-",  marker="^"),
    "kto":   dict(color="orchid",     linestyle="-",  marker="D"),
}


def main():
    parser = argparse.ArgumentParser(description="Plot cold-start results")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--out", default=None, help="Output directory (default: same as results)")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    cfg = data["config"]
    results = data["results"]
    eval_users = data["eval_users"]
    sizes = cfg["cold_start_sizes"]

    out_dir = Path(args.out) if args.out else Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert lists back to the expected format
    for m in results:
        results[m] = {int(k): v for k, v in results[m].items()}

    mean_res, se_res = compute_summary(results, sizes)

    # ── Main plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in mean_res:
        style = STYLES.get(name, {})
        ax.errorbar(sizes, mean_res[name], yerr=se_res[name],
                    label=name.upper(), capsize=3, **style)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(map(str, sizes))
    ax.set_xlabel("Training pairs per user (n)")
    ax.set_ylabel("Preference accuracy (mean ± SE)")
    ax.set_title(f"Cold Start: Preference Accuracy\n"
                 f"({len(eval_users)} users, {cfg['num_train_epochs']} epoch(s), "
                 f"LoRA r={cfg['lora_r']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "cold_start_results.png", dpi=150)
    print(f"Saved {out_dir / 'cold_start_results.png'}")

    # ── Per-user plot ─────────────────────────────────────────────────────────
    n_users = len(eval_users)
    ncols = min(n_users, 5)
    nrows = (n_users + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    if n_users == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, user in enumerate(eval_users):
        ax = axes[i]
        for name in STYLES:
            if name not in results:
                continue
            ax.plot(
                sizes,
                [results[name][nv][i] for nv in sizes],
                label=name.upper(),
                **STYLES[name],
            )
        ax.set_title(f"User {str(user)[:10]}", fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels(map(str, sizes), fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    axes[0].set_ylabel("Preference accuracy")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.suptitle(f"{cfg['dataset'].title()} Per-User Cold Start", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "cold_start_per_user.png", dpi=150)
    print(f"Saved {out_dir / 'cold_start_per_user.png'}")


if __name__ == "__main__":
    main()
