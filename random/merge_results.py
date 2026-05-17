"""
Merge a TASTE main 1results.json (n=3,5,10,50) with an n=100 follow-up 1results.json.

Usage:
    python merge_results.py --main outputs_from_gpu/goodreads/1results.json \\
                            --n100 outputs_from_gpu/goodreads_n100/1results.json \\
                            --out  outputs_from_gpu/goodreads/results.json

The script:
  1. Validates that methods and seeds match across the two files.
  2. Reports eval-user overlap between the two runs.
  3. Flags config discrepancies (learning_rate, pairs_per_user, icl_max_seq_len).
  4. Scans for acc=0.0 cells (likely truncation bugs) and reports them.
  5. Produces a merged 1results.json with sizes from both runs.
  6. Prints a summary table of mean ± SE across all sizes.

Per-user paired tests (Wilcoxon) within each n remain valid because the comparison
is WITHIN a single eval set. Cross-n comparison via the trajectory plot uses means,
which are valid across different eval sets as long as eval users are drawn from the
same population (book/movie reviewers from the same source dataset).
"""

import argparse
import json
import math
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def mean_se(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return m, se


def merge(main_path, n100_path, out_path):
    main = load(main_path)
    n100 = load(n100_path)

    print("=" * 70)
    print("CONFIG SANITY CHECKS")
    print("=" * 70)

    # Check methods match
    main_methods = set(main["results"].keys())
    n100_methods = set(n100["results"].keys())
    if main_methods != n100_methods:
        print(f"  ⚠ Method sets DIFFER:")
        print(f"    Only in main:  {main_methods - n100_methods}")
        print(f"    Only in n100:  {n100_methods - main_methods}")
    else:
        print(f"  ✓ Methods match: {sorted(main_methods)}")

    # Check seeds match
    main_seeds = main.get("seeds", [])
    n100_seeds = n100.get("seeds", [])
    if main_seeds != n100_seeds:
        print(f"  ⚠ Seeds differ! main={main_seeds}, n100={n100_seeds}")
    else:
        print(f"  ✓ Seeds match: {main_seeds}")

    # Config flags
    keys_to_check = [
        "learning_rate",
        "num_train_epochs",
        "batch_size",
        "lora_r",
        "lora_alpha",
        "max_seq_len",
        "icl_max_seq_len",
        "pairs_per_user",
        "dpo_beta",
        "simpo_beta",
        "kto_beta",
    ]
    differences = []
    for k in keys_to_check:
        v_main = main["config"].get(k)
        v_n100 = n100["config"].get(k)
        if v_main != v_n100:
            differences.append((k, v_main, v_n100))

    if differences:
        print("\n  ⚠ CONFIG DIFFERENCES DETECTED:")
        print(f"  {'key':<25} {'main':<15} {'n=100':<15}")
        for k, vm, vn in differences:
            print(f"  {k:<25} {str(vm):<15} {str(vn):<15}")
        print("\n  ⚠ These differences may affect the validity of n-trajectory comparisons.")
        print("  Document them in the thesis as caveats.")
    else:
        print("\n  ✓ All checked config keys match between runs.")

    # Eval user overlap
    print("\n" + "=" * 70)
    print("EVAL USER OVERLAP")
    print("=" * 70)
    main_users = set(main["eval_users"])
    n100_users = set(n100["eval_users"])
    overlap = main_users & n100_users
    print(f"  Main eval users:    {len(main_users)}")
    print(f"  n=100 eval users:   {len(n100_users)}")
    print(f"  Overlap:            {len(overlap)} users ({100*len(overlap)/len(main_users):.0f}%)")
    print(f"  Only in main:       {sorted(main_users - n100_users)}")
    print(f"  Only in n=100:      {sorted(n100_users - main_users)}")

    # Outlier scan
    print("\n" + "=" * 70)
    print("OUTLIER SCAN (acc = 0.0 cells, likely truncation bugs)")
    print("=" * 70)
    found_outliers = False
    for src_name, src in [("MAIN", main), ("N=100", n100)]:
        for m, sizes in src["results"].items():
            for n, seeds in sizes.items():
                for s, accs in seeds.items():
                    for i, acc in enumerate(accs):
                        if acc == 0.0 and m != "base":  # base near-chance is OK; 0.0 is suspicious
                            user_id = src["eval_users"][i] if i < len(src["eval_users"]) else "?"
                            print(f"  {src_name}: method={m} n={n} seed={s} user_idx={i} user_id={user_id}")
                            found_outliers = True
    if not found_outliers:
        print("  No 0.0 cells detected.")

    # Build merged result
    print("\n" + "=" * 70)
    print("MERGING")
    print("=" * 70)

    merged = json.loads(json.dumps(main))  # deep copy

    for method, sizes in n100["results"].items():
        if method not in merged["results"]:
            merged["results"][method] = {}
        for n, seed_data in sizes.items():
            if n in merged["results"][method]:
                print(f"  ⚠ Overwriting existing data for {method} at n={n}")
            merged["results"][method][n] = seed_data

    # Update sizes
    all_sizes = sorted(
        {int(s) for m in merged["results"].values() for s in m.keys()}
    )
    merged["config"]["cold_start_sizes"] = all_sizes

    # Add merge metadata
    merged["merge_metadata"] = {
        "main_source": str(main_path),
        "n100_source": str(n100_path),
        "main_eval_users": sorted(main_users),
        "n100_eval_users": sorted(n100_users),
        "eval_user_overlap_count": len(overlap),
        "main_eval_users_only": sorted(main_users - n100_users),
        "n100_eval_users_only": sorted(n100_users - main_users),
        "config_differences": [
            {"key": k, "main": vm, "n100": vn} for k, vm, vn in differences
        ],
    }

    print(f"  Merged sizes: {all_sizes}")
    print(f"  Methods: {sorted(merged['results'].keys())}")

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Saved to: {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("MERGED SUMMARY TABLE (mean ± SE across all users × seeds)")
    print("=" * 70)
    methods = list(merged["results"].keys())
    header = "  " + "method".ljust(12) + "".join(f"n={n:<14}" for n in all_sizes)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in methods:
        row = "  " + m.ljust(12)
        for n in all_sizes:
            seeds = merged["results"][m].get(str(n), {})
            all_acc = []
            for s, accs in seeds.items():
                all_acc.extend(accs)
            if all_acc:
                mu, se = mean_se(all_acc)
                row += f"{mu:.3f}±{se:.3f}  "
            else:
                row += " " * 16
        print(row)

    # ICL_CHAT outlier-excluded mean (for the user 11257 case at n=100)
    print("\n" + "=" * 70)
    print("ICL_CHAT n=100 SENSITIVITY (excluding acc=0.0 cells)")
    print("=" * 70)
    if "icl_chat" in merged["results"] and "100" in merged["results"]["icl_chat"]:
        all_acc = []
        all_acc_filtered = []
        for s, accs in merged["results"]["icl_chat"]["100"].items():
            all_acc.extend(accs)
            all_acc_filtered.extend([a for a in accs if a > 0])
        mu_full, se_full = mean_se(all_acc)
        mu_filt, se_filt = mean_se(all_acc_filtered)
        print(f"  With outliers:    {mu_full:.3f} ± {se_full:.3f}  (n={len(all_acc)})")
        print(f"  Without outliers: {mu_filt:.3f} ± {se_filt:.3f}  (n={len(all_acc_filtered)})")
        print(f"  → Outlier effect: drops mean by {mu_filt - mu_full:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True, help="Path to main 1results.json (n=3,5,10,50)")
    parser.add_argument("--n100", required=True, help="Path to n=100 1results.json")
    parser.add_argument("--out", default="results.json", help="Output merged 1results.json")
    args = parser.parse_args()
    merge(args.main, args.n100, args.out)