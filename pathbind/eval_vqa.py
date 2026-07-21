"""PathBind VQA evaluation: with-image vs text-only accuracy + Multimodal Gain.

Usage
-----
    python -m pathbind.eval_vqa \\
        --data path/to/pathbind_vqa_1500.tsv \\
        --model_module pathbind.models.patho_r1 \\
        --ckpt /path/to/Patho-R1 \\
        --out results/patho-r1_pathbind_vqa.jsonl

For PathBind-PTA, `--data path/to/pathbind_pta_600.tsv` — the loader
auto-detects that the `image` column contains base64 and decodes it in memory.
"""
from __future__ import annotations

import argparse
import base64
import csv
import importlib
import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

# PTA rows contain full base64 JPEGs (~250 KB each after downscale), well
# above csv's 128 KB default field limit. Raise once at import.
csv.field_size_limit(64 * 1024 * 1024)

from .prompts import build_mcq_messages, build_yn_messages, parse_answer


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def _decode_b64(s: str) -> Image.Image:
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def load_samples(tsv_path: Path) -> list[dict]:
    """Return sample dicts with:
        sample_id, question, options[], answer, answer_type,
        dimension, image  (PIL.Image OR path str)
    """
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    samples = []
    for r in rows:
        opts = [r[L] for L in "ABCDEFGHIJKLMNOP" if r.get(L, "")]
        # image: prefer base64 column, else filesystem path
        img_col = r.get("image", "") or ""
        if len(img_col) > 300:      # base64-embedded (PathBind-PTA)
            image: Any = _decode_b64(img_col)
        else:
            image = r.get("image_path", "")
        samples.append({
            "sample_id": r["index"],
            "question": r["question"],
            "options": opts,
            "answer": r["answer"].strip(),
            "answer_type": r.get("answer_type", "mcq"),
            "dimension": r.get("dimension", ""),
            "image": image,
        })
    return samples


# ------------------------------------------------------------------ #
# Per-sample eval
# ------------------------------------------------------------------ #
def build_messages(sample: dict, condition: str):
    image = None if condition == "text_only" else sample["image"]
    if sample["answer_type"] == "mcq":
        return build_mcq_messages(sample["question"], sample["options"], image)
    return build_yn_messages(sample["question"], image)


def score(pred: str | None, gt: str, atype: str) -> bool:
    if pred is None:
        return False
    return pred.upper() == gt.upper() if atype == "mcq" else pred.lower() == gt.lower()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True,
                   help="Path to a PathBind TSV (VQA-1500 or PTA-600)")
    p.add_argument("--model_module", type=str, required=True,
                   help="Python module implementing `load_model(ckpt, gpus)` "
                        "(e.g. pathbind.models.patho_r1)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path passed to load_model (optional)")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--conditions", nargs="+",
                   default=["with_image", "text_only"],
                   choices=["with_image", "text_only"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max_new", type=int, default=768)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    print(f"[vqa] loading model from {args.model_module}", flush=True)
    mod = importlib.import_module(args.model_module)
    load = getattr(mod, "load_model")
    model = load(args.ckpt, gpus=args.gpus) if args.ckpt else load(gpus=args.gpus)

    print(f"[vqa] loading data from {args.data}", flush=True)
    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[vqa] {len(samples)} samples, conditions={args.conditions}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with args.out.open("w") as f:
        for s in samples:
            rec = {
                "sample_id": s["sample_id"],
                "dimension": s["dimension"],
                "answer_type": s["answer_type"],
                "gt": s["answer"].upper() if s["answer_type"] == "mcq"
                        else s["answer"].lower(),
            }
            for cond in args.conditions:
                t0 = time.time()
                try:
                    raw = model.generate(build_messages(s, cond),
                                          max_new=args.max_new)
                except Exception as e:
                    rec[cond] = {"raw": "", "error": str(e), "elapsed_sec": 0}
                    continue
                pred = parse_answer(raw, s["answer_type"])
                rec[cond] = {
                    "raw": raw,
                    "letter": pred,
                    "correct": score(pred, rec["gt"], s["answer_type"]),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done <= 3 or n_done % 50 == 0:
                w = rec.get("with_image", {}); t = rec.get("text_only", {})
                print(f"[{n_done:04d}/{len(samples)}] gt={rec['gt']}  "
                      f"img={w.get('letter')}{'✓' if w.get('correct') else '✗'}  "
                      f"txt={t.get('letter')}{'✓' if t.get('correct') else '✗'}",
                      flush=True)

    # Aggregate
    print(f"[vqa] DONE: {args.out}", flush=True)
    with args.out.open() as f:
        rows = [json.loads(l) for l in f]
    for cond in args.conditions:
        n_valid = sum(1 for r in rows if (r.get(cond) or {}).get("letter") is not None)
        n_correct = sum(1 for r in rows if (r.get(cond) or {}).get("correct"))
        acc = n_correct / n_valid * 100 if n_valid else 0
        print(f"  Acc({cond}) = {acc:.2f}%  (n={n_valid})")
    if len(args.conditions) == 2:
        w_c = sum(1 for r in rows if (r.get("with_image") or {}).get("correct"))
        w_n = sum(1 for r in rows if (r.get("with_image") or {}).get("letter") is not None)
        t_c = sum(1 for r in rows if (r.get("text_only") or {}).get("correct"))
        t_n = sum(1 for r in rows if (r.get("text_only") or {}).get("letter") is not None)
        if w_n and t_n:
            print(f"  MG (Acc_img - Acc_text) = "
                  f"{w_c/w_n*100 - t_c/t_n*100:+.2f} pp")


if __name__ == "__main__":
    main()
