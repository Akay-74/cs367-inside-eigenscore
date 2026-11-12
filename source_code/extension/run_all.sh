#!/usr/bin/env bash
# Full extension runs: 1,000 questions each on CoQA, TriviaQA, NQ (resumable).
cd "$(dirname "$0")"
PY="${PY:-/home/akay/Desktop/Holy Research/eigenscore/.venv/bin/python}"
COQA="${COQA:-/home/akay/Desktop/Holy Research/eigenscore/data/datasets/coqa-dev-v1.0.json}"
for ds in coqa triviaqa nq; do
  echo "=== $ds start $(date)" >> results/run_all.log
  "$PY" -u run_extension.py --dataset $ds --n 1000 --resume --coqa_json "$COQA" \
      --out results/llama2_$ds >> results/$ds.log 2>&1
  echo "=== $ds exit=$? $(date)" >> results/run_all.log
done
