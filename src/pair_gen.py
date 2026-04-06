"""Generate DPO and KTO preference pairs from a ratings DataFrame."""

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def build_preference_pools(df: pd.DataFrame):
    """Split each user's items into liked (>=4) and disliked (<=2) pools.

    Returns (liked_by_user, disliked_by_user) dicts mapping user_id → list[title].
    Ambiguous titles appearing in both pools for the same user are removed.
    """
    liked = (
        df[df["rating"] >= 4]
        .groupby("user_id")["title"]
        .apply(lambda x: sorted(set(x)))
    )
    disliked = (
        df[df["rating"] <= 2]
        .groupby("user_id")["title"]
        .apply(lambda x: sorted(set(x)))
    )

    overlap_removed = 0
    for user in liked.index.intersection(disliked.index):
        overlap = set(liked[user]) & set(disliked[user])
        if overlap:
            overlap_removed += len(overlap)
            liked[user] = [t for t in liked[user] if t not in overlap]
            disliked[user] = [t for t in disliked[user] if t not in overlap]

    if overlap_removed:
        print(f"  Removed {overlap_removed:,} ambiguous titles")

    return liked, disliked


def select_users(liked, disliked, n_users: int, pairs_per_user: int):
    """Select the n_users with the highest pairing capacity."""
    eligible = [
        u
        for u in liked.index.intersection(disliked.index)
        if len(liked[u]) >= 2 and len(disliked[u]) >= 2
    ]
    capacity = {u: len(liked[u]) * len(disliked[u]) for u in eligible}
    eligible = [u for u in eligible if capacity[u] >= pairs_per_user]
    print(f"  Eligible users (capacity >= {pairs_per_user}): {len(eligible):,}")

    if len(eligible) < n_users:
        n_users = len(eligible)
        print(f"  Reduced n_users to {n_users}")

    pool = sorted(eligible, key=lambda u: capacity[u], reverse=True)[:n_users]
    print(f"  Selected {len(pool)} users")
    return pool


def generate_pairs(liked, disliked, user_pool, pairs_per_user: int, seed: int = 42):
    """Sample (liked, disliked) pairs per user. Returns list of tuples."""
    rng = random.Random(seed)
    all_pairs = []

    for user in user_pool:
        user_pairs, seen, attempts = [], set(), 0
        while len(user_pairs) < pairs_per_user and attempts < pairs_per_user * 20:
            lt = rng.choice(liked[user])
            dt = rng.choice(disliked[user])
            attempts += 1
            key = (lt, dt)
            if lt == dt or key in seen:
                continue
            seen.add(key)
            pair_id = f"{user}_{len(user_pairs)}"
            user_pairs.append((user, lt, dt, pair_id))
        all_pairs.extend(user_pairs)

    rng.shuffle(all_pairs)
    counts = Counter(p[0] for p in all_pairs)
    print(f"  Total pairs: {len(all_pairs):,}  |  Users: {len(counts)}  |  "
          f"Per-user: {min(counts.values())}–{max(counts.values())}")
    return all_pairs


def write_dpo_jsonl(pairs, output_path: str, prompt_template: str, seed: int = 123):
    """Write DPO JSONL with randomised A/B positioning."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coin = random.Random(seed)

    with open(output_path, "w") as f:
        for user, liked, disliked, pair_id in pairs:
            if coin.random() < 0.5:
                title_a, title_b = liked, disliked
            else:
                title_a, title_b = disliked, liked
            f.write(
                json.dumps({
                    "pair_id": pair_id,
                    "user_id": user,
                    "prompt": prompt_template.format(title_a=title_a, title_b=title_b),
                    "chosen": f"Approved: {liked}, Denied: {disliked}",
                    "rejected": f"Approved: {disliked}, Denied: {liked}",
                }) + "\n"
            )
    print(f"  DPO → {output_path}  ({len(pairs):,} rows)")


def write_kto_jsonl(pairs, output_path: str, prompt_template: str):
    """Write KTO JSONL (one pos + one neg row per pair)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for user, liked, disliked, pair_id in pairs:
            pair_rng = random.Random(hash(("kto_coin", pair_id)))
            if pair_rng.random() < 0.5:
                title_a, title_b = liked, disliked
            else:
                title_a, title_b = disliked, liked
            prompt = prompt_template.format(title_a=title_a, title_b=title_b)

            f.write(json.dumps({
                "pair_id": pair_id, "user_id": user, "prompt": prompt,
                "completion": f"Approved: {liked}, Denied: {disliked}",
                "label": True,
            }) + "\n")
            f.write(json.dumps({
                "pair_id": pair_id, "user_id": user, "prompt": prompt,
                "completion": f"Approved: {disliked}, Denied: {liked}",
                "label": False,
            }) + "\n")
    print(f"  KTO → {output_path}  ({len(pairs) * 2:,} rows)")
