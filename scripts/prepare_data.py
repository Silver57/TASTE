#!/usr/bin/env python3
"""
Step 1: Load raw data → generate DPO + KTO preference pairs.

Usage:
    python scripts/prepare_data.py --config configs/20_05_goodreads_recsys.yaml
    python scripts/prepare_data.py --config configs/20_05_goodreads_recsys.yaml --ablate-prompts
    python scripts/prepare_data.py --config configs/20_05_goodreads_recsys.yaml --smoke-test
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pair_gen import (
    build_preference_pools,
    generate_pairs,
    select_users,
    write_dpo_jsonl,
    write_kto_jsonl,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def resolve_templates(cfg, ablate: bool) -> dict[str, str]:
    """Return {name: template_string} dict.

    Without --ablate-prompts: only the default template.
    With --ablate-prompts: all templates.
    Backwards-compatible with old configs that use `prompt_template`.
    """
    if "prompt_templates" in cfg:
        templates = cfg["prompt_templates"]
        default = cfg.get("default_prompt", next(iter(templates)))
        if ablate:
            return templates
        return {default: templates[default]}

    # Legacy single-template config
    return {"default": cfg["prompt_template"]}


def main():
    parser = argparse.ArgumentParser(description="Generate preference pairs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--ablate-prompts", action="store_true",
        help="Generate data for ALL prompt templates (default: only the default one)",
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

    cfg = load_config(config_path)
    random.seed(cfg.get("random_seed", 42))

    dataset = cfg["dataset"]
    data_dir = cfg["data_dir"]
    pair_dir = Path(cfg["pair_dir"])

    # Load dataset
    if dataset == "netflix":
        from src.data_netflix import load_netflix
        df = load_netflix(data_dir, min_reviews=cfg.get("min_reviews_per_movie", 20000))
    elif dataset == "20_05_goodreads_recsys":
        from src.data_goodreads import load_goodreads
        df = load_goodreads(data_dir, min_reviews=cfg.get("min_reviews_per_book", 20000))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Build pools and generate pairs
    liked, disliked = build_preference_pools(df)
    user_pool = select_users(
        liked, disliked,
        n_users=cfg["n_users"],
        pairs_per_user=cfg["pairs_per_user"],
    )
    pairs = generate_pairs(
        liked, disliked, user_pool,
        pairs_per_user=cfg["pairs_per_user"],
        seed=cfg.get("random_seed", 42),
    )

    # Write JSONL files — one set per template
    templates = resolve_templates(cfg, args.ablate_prompts)
    for name, template in templates.items():
        print(f"\n── Prompt variant: {name!r} ──")
        if len(templates) == 1:
            # Single template → write to the root pair_dir (same as before)
            out = pair_dir
        else:
            out = pair_dir / name
        write_dpo_jsonl(pairs, out / "dpo_pairs.jsonl", template)
        write_kto_jsonl(pairs, out / "kto_examples.jsonl", template)

    print("\nDone.")


if __name__ == "__main__":
    main()
