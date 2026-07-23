#!/bin/bash
# Reference launch script: run Patho-R1-7B on all three PathBind components.
# Edit CKPT and paths to your local layout.

set -e
CKPT=${CKPT:-./Patho-R1}
GPUS=${GPUS:-0}
OUT_DIR=${OUT_DIR:-results/patho-r1}
mkdir -p "$OUT_DIR"

# ---- VQA on PathBind-VQA (build.py must have been run first) ----
python -m pathbind.eval_vqa \
    --data pathbind/data/PathBind-VQA/pathbind_vqa_1500.tsv \
    --model_module pathbind.models.patho_r1 --ckpt "$CKPT" --gpus "$GPUS" \
    --out "$OUT_DIR/pathbind_vqa.jsonl"

# ---- VQA on PathBind-PTA (base64 images embedded) ----
python -m pathbind.eval_vqa \
    --data pathbind/data/PathBind-PTA/pathbind_pta_600.tsv \
    --model_module pathbind.models.patho_r1 --ckpt "$CKPT" --gpus "$GPUS" \
    --out "$OUT_DIR/pathbind_pta.jsonl"

# ---- Grounding on PathBind-Grounding (build.py must have been run first) ----
python -m pathbind.eval_grounding \
    --data pathbind/data/PathBind-Grounding/pathbind_grounding_500.jsonl \
    --model_module pathbind.models.patho_r1 --ckpt "$CKPT" --gpus "$GPUS" \
    --attn_impl sdpa \
    --out "$OUT_DIR/pathbind_grounding.jsonl"
