# Cold-Start Preference Learning — Master's Thesis

Compares DPO, IPO, SimPO, and KTO for personalised cold-start recommendation
using LoRA-adapted LLMs. Evaluates how quickly each alignment method recovers
user preferences from minimal interaction data.

## Project layout

```
cold_start_thesis/
├── configs/
│   ├── netflix.yaml          # Netflix experiment config
│   └── goodreads.yaml        # GoodReads experiment config
├── src/
│   ├── __init__.py
│   ├── device.py             # Device / dtype auto-detection
│   ├── data_netflix.py       # Netflix data loading & EDA
│   ├── data_goodreads.py     # GoodReads data loading
│   ├── pair_gen.py           # Preference-pair generation (DPO + KTO JSONL)
│   ├── logprobs.py           # Completion-only log-prob scoring
│   ├── trainers.py           # Custom DPO / IPO / SimPO / KTO trainers
│   └── evaluate.py           # Preference accuracy + statistical tests
├── scripts/
│   ├── prepare_data.py       # Step 1: parse raw data → preference pairs
│   ├── run_sweep.py          # Step 2: cold-start training sweep
│   └── plot_results.py       # Step 3: figures from saved results
├── data/                     # Created at runtime (git-ignored)
├── requirements.txt
├── setup.sh                  # One-command vast.ai bootstrap
└── README.md
```

## Quick start (vast.ai / any GPU box)

```bash
# 1. SSH into your instance, clone or upload this folder, then:
cd cold_start_thesis
bash setup.sh               # installs deps + logs into HF

# 2. Place raw data under data/raw/
#    Netflix:   data/raw/Netflix/combined_data_*.txt + movie_titles.csv
#    GoodReads: data/raw/GoodReads/ratings.csv + books.csv

# 3. Generate preference pairs
python scripts/prepare_data.py --config configs/goodreads.yaml

# 4. Run the cold-start sweep
python scripts/run_sweep.py --config configs/goodreads.yaml

# 5. Plot results
python scripts/plot_results.py --results outputs/goodreads/results.json
```

## Adding a new dataset

1. Write a loader in `src/data_<name>.py` that returns a DataFrame with
   columns `user_id`, `item_id`, `title`, `rating`.
2. Add a YAML config under `configs/<name>.yaml`.
3. Register the loader in `scripts/prepare_data.py` (one `elif` block).

## Adding a new alignment method

1. Subclass `_BaseTrainer` in `src/trainers.py`.
2. Add the method name to the `METHODS` list in `scripts/run_sweep.py`.
3. That's it — the sweep loop picks it up automatically.

## Configs

All hyperparameters live in YAML files so experiments are reproducible
without editing code. Override any value from the CLI:

```bash
python scripts/run_sweep.py --config configs/goodreads.yaml \
    --cold_start_sizes 3 5 10 20 50 \
    --n_eval_users 10 \
    --num_train_epochs 3
```
