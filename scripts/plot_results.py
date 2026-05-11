#!/usr/bin/env python3
"""
Step 3: Generate figures from saved results.

Usage:
    # Single result file (unchanged behaviour)
    python scripts/plot_results.py --results outputs/goodreads/results.json

    # Prompt ablation — pass every variant's file
    python scripts/plot_results.py --results outputs/goodreads/results_simple.json \
                                              outputs/goodreads/results_detailed.json \
                                              outputs/goodreads/results_system.json

    # Or just point at the directory and it picks up all results_*.json
    python scripts/plot_results.py --results-dir outputs/goodreads/

    # Custom output directory
    python scripts/plot_results.py --results-dir outputs/goodreads/ --out figures/
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
    "base":     dict(color="gray",         linestyle="--", marker=""),
    "dpo":      dict(color="steelblue",    linestyle="-",  marker="o"),
    "ipo":      dict(color="darkorange",   linestyle="-",  marker="s"),
    "simpo":    dict(color="seagreen",     linestyle="-",  marker="^"),
    "kto":      dict(color="orchid",       linestyle="-",  marker="D"),
    "icl_flat": dict(color="crimson",      linestyle=":",  marker="*"),
    "icl_chat": dict(color="mediumpurple", linestyle=":",  marker="X"),
}

# Extra linestyles cycled across prompt variants in the ablation comparison plot
_VARIANT_LS = ["-", "--", "-.", ":"]


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _fix_keys(results: dict) -> dict:
    """Convert string keys back to ints."""
    return {m: {int(k): v for k, v in results[m].items()} for m in results}


# ── Single-file plots (original behaviour) ──────────────────────────────────

def plot_single(data: dict, out_dir: Path):
    cfg = data["config"]
    results = _fix_keys(data["results"])
    eval_users = data["eval_users"]
    sizes = cfg["cold_start_sizes"]
    prompt_label = data.get("prompt_name") or ""

    mean_res, se_res = compute_summary(results, sizes)

    # Main plot
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
    title = (f"Cold Start: Preference Accuracy\n"
             f"({len(eval_users)} users, {cfg['num_train_epochs']} epoch(s), "
             f"LoRA r={cfg['lora_r']})")
    if prompt_label:
        title += f"\nprompt: {prompt_label!r}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    suffix = f"_{prompt_label}" if prompt_label else ""
    fname = out_dir / f"cold_start_results{suffix}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

    # Per-user plot
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
    fig.suptitle(f"{cfg['dataset'].title()} Per-User Cold Start"
                 + (f" — {prompt_label!r}" if prompt_label else ""),
                 fontsize=11)
    plt.tight_layout()
    fname = out_dir / f"cold_start_per_user{suffix}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# ── Ablation comparison plot ─────────────────────────────────────────────────

def plot_ablation(all_data: dict[str, dict], out_dir: Path):
    """One figure comparing all prompt variants side by side per method."""
    # Use first variant to get shared config values
    first = next(iter(all_data.values()))
    cfg = first["config"]
    sizes = cfg["cold_start_sizes"]
    methods = [m for m in first["results"] if m != "base"]
    variant_names = list(all_data.keys())

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        for vi, vname in enumerate(variant_names):
            results = _fix_keys(all_data[vname]["results"])
            mean_res, se_res = compute_summary(results, sizes)

            ls = _VARIANT_LS[vi % len(_VARIANT_LS)]
            color = STYLES.get(method, {}).get("color", None)
            ax.errorbar(
                sizes, mean_res[method], yerr=se_res[method],
                label=vname, capsize=3, linestyle=ls, color=color,
                marker=STYLES.get(method, {}).get("marker", "o"),
                alpha=0.6 + 0.4 * (vi == 0),
            )

        # Also plot base (same across variants) once
        base_results = _fix_keys(all_data[variant_names[0]]["results"])
        base_mean, base_se = compute_summary(base_results, sizes)
        ax.errorbar(sizes, base_mean["base"], yerr=base_se["base"],
                    label="base", capsize=3, **STYLES["base"])

        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels(map(str, sizes))
        ax.set_xlabel("Training pairs (n)")
        ax.set_title(method.upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Preference accuracy (mean ± SE)")
    fig.suptitle(f"{cfg['dataset'].title()} — Prompt Ablation", fontsize=13)
    plt.tight_layout()
    fname = out_dir / "prompt_ablation.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

    # Summary table: best method per variant
    print(f"\n{'─' * 50}")
    print("Prompt ablation summary (accuracy at largest n):")
    print(f"{'─' * 50}")
    largest = sizes[-1]
    header = f"{'Variant':<15}" + "".join(f"{m.upper():>10}" for m in methods)
    print(header)
    for vname in variant_names:
        results = _fix_keys(all_data[vname]["results"])
        means, _ = compute_summary(results, sizes)
        vals = "".join(f"{means[m][-1]:>10.3f}" for m in methods)
        print(f"{vname:<15}{vals}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot cold-start results")
    parser.add_argument(
        "--results", nargs="*", default=None,
        help="Path(s) to results JSON file(s)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory to scan for results*.json (alternative to --results)",
    )
    parser.add_argument("--out", default=None, help="Output directory (default: same as results)")
    args = parser.parse_args()

    # Collect result files
    if args.results_dir:
        rdir = Path(args.results_dir)
        files = sorted(rdir.glob("results*.json"))
        if not files:
            parser.error(f"No results*.json found in {rdir}")
        # If ablation files exist (results_<name>.json), drop the plain
        # results.json to avoid a duplicate "default" vs the real variant.
        ablation_files = [f for f in files if f.name != "results.json"]
        if ablation_files:
            files = ablation_files
    elif args.results:
        files = [Path(p) for p in args.results]
    else:
        parser.error("Provide --results or --results-dir")

    out_dir = Path(args.out) if args.out else files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all files
    all_data = {}
    for f in files:
        data = load_result(f)
        label = data.get("prompt_name") or f.stem.removeprefix("results").lstrip("_") or "default"
        all_data[label] = data

    # Always produce per-file plots
    for label, data in all_data.items():
        plot_single(data, out_dir)

    # If multiple variants, also produce the comparison plot
    if len(all_data) > 1:
        plot_ablation(all_data, out_dir)


if __name__ == "__main__":
    main()
