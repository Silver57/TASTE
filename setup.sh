#!/usr/bin/env bash
# setup.sh — bootstrap a vast.ai (or any Ubuntu+CUDA) instance
set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Hugging Face login ==="
echo "You need a HF token with access to Meta-Llama-3-8B-Instruct."
echo "Get one at https://huggingface.co/settings/tokens"
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    huggingface-cli login
else
    huggingface-cli login --token "$HF_TOKEN"
    echo "Logged in via HF_TOKEN env var."
fi

mkdir -p data/raw outputs

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Place raw data under data/raw/  (Netflix/ or GoodReads/)"
echo "  2. python scripts/prepare_data.py --config configs/goodreads.yaml"
echo "  3. python scripts/run_sweep.py    --config configs/goodreads.yaml"
