#!/usr/bin/env bash
# Run the full pilot pipeline end-to-end.
# Usage: bash run_all.sh [--max-pairs 200]
#
# Prerequisites:
#   pip install -r requirements.txt
#   cp .env.example .env && edit .env with your OPENAI_API_KEY

set -e

MAX_PAIRS=${1:-""}  # pass --max-pairs 200 for a quick test run

echo "=== Step 1: Download data ==="
python 01_download_data.py

echo ""
echo "=== Step 2: 8-gram baseline (expect ~0 hits) ==="
python 02_ngram_baseline.py

echo ""
echo "=== Step 3A: Embedding retrieval ==="
python 03_embedding_retrieval.py

echo ""
echo "=== Step 3B: LLM judge ==="
if [ -n "$MAX_PAIRS" ]; then
    python 04_llm_judge.py --max-pairs "$MAX_PAIRS" --resume
else
    python 04_llm_judge.py --resume
fi

echo ""
echo "=== Step 4: Analyze results ==="
python 05_analyze_results.py

echo ""
echo "=== Done! ==="
echo "Key outputs:"
echo "  results/pilot_report.md       — summary with the 4 key numbers"
echo "  results/flagged_pairs.tsv     — all CONTAMINATED/RELATED pairs"
echo "  results/precision_worksheet.tsv — fill this in for manual validation"
