#!/usr/bin/env bash
# INSIDE / EigenScore -- generation + evaluation on CoQA.
#
# Tuned for an 8GB GPU: 4-bit NF4 weights, K=10 generations split across two
# forward passes of 5. Drop --load_in_4bit on a 24GB+ card to match the paper's
# fp16 setup, and raise --fraction_of_data_to_use to 1 for the full 7983-pair split.
set -euo pipefail

MODEL=${MODEL:-llama2-7b-hf}
DATASET=${DATASET:-coqa}
FRACTION=${FRACTION:-0.04}
PROJECT_IND=${PROJECT_IND:-0}

python -m pipeline.generate \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --load_in_4bit --low_memory \
    --feature_clip \
    --num_generations_per_prompt 10 \
    --num_return_sequences 5 \
    --fraction_of_data_to_use "$FRACTION" \
    --project_ind "$PROJECT_IND"

python -m func.evalFunc \
    "data/output/${MODEL}_${DATASET}_${PROJECT_IND}/0.pkl" \
    --correctness rouge
