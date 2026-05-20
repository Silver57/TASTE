#!/usr/bin/env python3
"""
Merge classical recommender results into the matching LLM results.json.

For each results_classical[_<prompt>].json in the target directory, finds the
companion results[_<prompt>].json, asserts eval_users / seeds match
positionally, then copies the 'pop', 'iknn', 'bpr' method dicts into the LLM
results file in place (writing a .bak alongside first).

Usage:
    python scripts/merge_classical.py --dir outputs/goodreads_smoke
    python scripts/merge_classical.py --dir outputs/20_05_goodreads_recsys
"""

import argparse
import json
import shutil
from pathlib import Path

CLASSICAL_METHODS = ("pop", "iknn", "bpr")


def _suffix_from_classical(p: Path) -> str:
    """results_classical.json → '', results_classical_simple.json → '_simple'."""
    stem = p.stem  # 'results_classical' or 'results_classical_<name>'
    assert stem.startswith("results_classical"), p
    rest = stem[len("results_classical"):]
    return rest  # '' or '_<name>'


def _llm_path_for(classical_path: Path) -> Path:
    suffix = _suffix_from_classical(classical_path)
    return classical_path.parent / f"results{suffix}.json"


def merge_one(classical_path: Path) -> None:
    llm_path = _llm_path_for(classical_path)
    if not llm_path.exists():
        raise FileNotFoundError(
            f"Companion LLM results not found: {llm_path}\n"
            f"(for classical file: {classical_path})"
        )

    with open(classical_path) as f:
        classical = json.load(f)
    with open(llm_path) as f:
        llm = json.load(f)

    if classical["eval_users"] != llm["eval_users"]:
        raise AssertionError(
            f"eval_users mismatch between {classical_path} and {llm_path}.\n"
            f"  classical: {classical['eval_users']}\n"
            f"  llm:       {llm['eval_users']}\n"
            f"Per-user accuracy lists are positional — refusing to merge."
        )
    # Compare seeds as lists of ints regardless of source type
    if [int(s) for s in classical["seeds"]] != [int(s) for s in llm["seeds"]]:
        raise AssertionError(
            f"seeds mismatch between {classical_path} and {llm_path}.\n"
            f"  classical: {classical['seeds']}\n"
            f"  llm:       {llm['seeds']}"
        )

    # Backup the LLM file once per merge invocation (overwrites prior .bak)
    bak = llm_path.with_suffix(llm_path.suffix + ".bak")
    shutil.copy2(llm_path, bak)

    merged = 0
    for m in CLASSICAL_METHODS:
        cell = classical["results"].get(m)
        if cell is None:
            continue
        llm["results"][m] = cell
        merged += 1

    with open(llm_path, "w") as f:
        json.dump(llm, f, indent=2, default=str)

    print(f"  {classical_path.name} → {llm_path.name}  "
          f"[{merged} methods merged, backup: {bak.name}]")


def main():
    parser = argparse.ArgumentParser(description="Merge classical results into LLM results")
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing results*.json and results_classical*.json",
    )
    args = parser.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        parser.error(f"Not a directory: {d}")

    classical_files = sorted(d.glob("results_classical*.json"))
    if not classical_files:
        parser.error(f"No results_classical*.json in {d}")

    print(f"Merging classical results in {d} …")
    for p in classical_files:
        merge_one(p)
    print("Done.")


if __name__ == "__main__":
    main()
