# Cold-Start Preference Learning — Master's Thesis

Compares DPO, IPO, SimPO, and KTO for personalised cold-start recommendation
using LoRA-adapted LLMs, plus an in-context-learning baseline (`icl_flat` and
`icl_chat`) that pastes the n training pairs into the prompt and scores with
the frozen base model. Evaluates how quickly each alignment method recovers
user preferences from minimal interaction data, and whether training is worth
it at small n.

## Project layout

```
cold_start_thesis/
├── configs/
│   ├── netflix.yaml          # Netflix experiment config
│   ├── netflix_smoke.yaml    # Netflix smoke-test config (tiny local run)
│   ├── goodreads.yaml        # GoodReads experiment config
│   └── goodreads_smoke.yaml  # GoodReads smoke-test config (tiny local run)
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
python scripts/prepare_data.py --config configs/20_05_goodreads_recsys.yaml

# 4. Run the cold-start sweep
python scripts/run_sweep.py --config configs/20_05_goodreads_recsys.yaml

# 5. Plot results
python scripts/plot_results.py --results outputs/20_05_goodreads_recsys/1results.json
```

## Prompt template ablation

Each config defines multiple named prompt templates under `prompt_templates` with
a `default_prompt` key that selects which one runs normally:

```yaml
prompt_templates:
  simple: "Which book does the reader prefer: {title_a} or {title_b}?"
  detailed: "Given these two books, which one would the reader enjoy more? ..."
  system: "You are a book recommendation assistant. ..."
default_prompt: simple
```

By default only the `default_prompt` variant is used. Add `--ablate-prompts` to
run **all** variants:

```bash
# Generate pairs for every template
python scripts/prepare_data.py --config configs/20_05_goodreads_recsys.yaml --ablate-prompts

# Train + evaluate every template
python scripts/run_sweep.py --config configs/20_05_goodreads_recsys.yaml --ablate-prompts
```

Results are saved as `results_<name>.json` (e.g. `results_simple.json`,
`results_detailed.json`). To plot them:

```bash
# Point at the output directory — picks up all results_*.json automatically
python scripts/plot_results.py --results-dir outputs/20_05_goodreads_recsys/

# Or list files explicitly
python scripts/plot_results.py --results outputs/20_05_goodreads_recsys/results_simple.json \
                                          outputs/20_05_goodreads_recsys/results_detailed.json
```

This produces:
- Per-variant plots (`cold_start_results_simple.png`, etc.)
- A combined **`prompt_ablation.png`** comparing all variants side by side per method

## Smoke testing (local)

Use `--smoke-test` to swap any config for its `*_smoke.yaml` counterpart — tiny
user counts, a single cold-start size, and smaller LoRA rank so you can verify
the pipeline end-to-end on a laptop:

```bash
python scripts/prepare_data.py --config configs/20_05_goodreads_recsys.yaml --smoke-test
python scripts/run_sweep.py   --config configs/20_05_goodreads_recsys.yaml --smoke-test
```

This automatically loads `configs/goodreads_smoke.yaml` instead. Both flags
compose: `--smoke-test --ablate-prompts` runs all prompt variants with the tiny
smoke config.

## Adding a new dataset

1. Write a loader in `src/data_<name>.py` that returns a DataFrame with
   columns `user_id`, `item_id`, `title`, `rating`.
2. Add a YAML config under `configs/<name>.yaml`.
3. Register the loader in `scripts/prepare_data.py` (one `elif` block).

## Adding a new alignment method

1. Subclass `_BaseTrainer` in `src/trainers.py` and register it in
   `TRAINER_REGISTRY`.
2. Add the method name to the `methods` list in `scripts/run_sweep.py` and
   append a tuple to `method_specs` with the trainer's kwargs, training set,
   eval set, and KTO flag.
3. For no-train methods (e.g. ICL), override `eval_dataset(self, ev_ds)` to
   transform the eval rows and set `self.eval_max_length` to a per-method
   sequence-length cap. The sweep loop calls these hooks for every method.
4. Add a `STYLES` entry in `scripts/plot_results.py` so the curve shows up in
   figures.

## Configs

All hyperparameters live in YAML files so experiments are reproducible
without editing code. Override any value from the CLI:

```bash
python scripts/run_sweep.py --config configs/20_05_goodreads_recsys.yaml \
    --cold_start_sizes 3 5 10 20 50 \
    --n_eval_users 10 \
    --num_train_epochs 3
```
