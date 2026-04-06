#!/usr/bin/env python3
"""
Step 1: Load raw data → generate DPO + KTO preference pairs.

Usage:
    python scripts/prepare_data.py --config configs/goodreads.yaml
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


def main():
    parser = argparse.ArgumentParser(description="Generate preference pairs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.get("random_seed", 42))

    dataset = cfg["dataset"]
    data_dir = cfg["data_dir"]
    pair_dir = Path(cfg["pair_dir"])

    # Load dataset
    if dataset == "netflix":
        from src.data_netflix import load_netflix
        df = load_netflix(data_dir, min_reviews=cfg.get("min_reviews_per_movie", 20000))
    elif dataset == "goodreads":
        from src.data_goodreads import load_goodreads
        df = load_goodreads(data_dir)
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

    # Write JSONL files
    template = cfg["prompt_template"]
    write_dpo_jsonl(pairs, pair_dir / "dpo_pairs.jsonl", template)
    write_kto_jsonl(pairs, pair_dir / "kto_examples.jsonl", template)

    print("\nDone.")


if __name__ == "__main__":
    main()
