"""PathBind grounding evaluation: entity-token attention → IoU / Precision /
Recall on the 500-sample PathBind-Grounding set.

Protocol
--------
    1. Feed the model a minimal prompt: "Look at the image and locate: {entity}"
    2. Forward pass with output_attentions=True (needs eager/sdpa attn).
    3. Locate the entity tokens in the input sequence and average their
       attention over the middle-third of decoder layers (head-mean).
    4. Reshape to the model's patch grid (Hp × Wp), Gaussian smooth (σ=3),
       min-max normalize, upsample to image resolution, threshold at the
       top 20% quantile → binary attended mask.
    5. Compare mask to the GT bounding box (converted to a binary region).

Usage
-----
    python -m pathbind.eval_grounding \\
        --data path/to/pathbind_grounding_500.jsonl \\
        --model_module pathbind.models.patho_r1 \\
        --ckpt /path/to/Patho-R1 \\
        --out results/patho-r1_pathbind_grounding.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .prompts import ATTN_PROMPT


# ------------------------------------------------------------------ #
# Attention-map utilities (paper appendix C.1–C.2)
# ------------------------------------------------------------------ #
def _mid_third_layers(n_layers: int) -> list[int]:
    lo = n_layers // 3
    hi = (2 * n_layers) // 3
    return list(range(lo, hi))


def _find_entity_token_positions(input_ids, tokenizer, entity: str,
                                  start_idx: int = 0) -> list[int]:
    """Return positions of tokens whose decoded text overlaps `entity`."""
    entity = entity.lower().strip()
    if not entity:
        return []
    words = set(w.strip(".,;:!?()[]") for w in entity.split() if len(w) > 2)
    ids = input_ids[0].tolist()
    positions = []
    for i in range(start_idx, len(ids)):
        tok = tokenizer.decode([ids[i]]).lower().strip()
        if not tok or len(tok) < 2:
            continue
        if any(w in tok or tok in w for w in words):
            positions.append(i)
    return positions


def _heatmap_to_mask(heat_2d: np.ndarray, target_hw: tuple[int, int],
                     sigma: float = 3.0, top_pct: float = 0.20):
    """Smooth + upsample + threshold to a binary attended mask at image res."""
    heat = gaussian_filter(heat_2d.astype(np.float32), sigma=sigma)
    heat -= heat.min()
    if heat.max() > 0:
        heat /= heat.max()
    # Bilinear upsample to (H, W)
    H, W = target_hw
    ph, pw = heat.shape
    ys = np.linspace(0, ph - 1, H)
    xs = np.linspace(0, pw - 1, W)
    y0, y1 = np.floor(ys).astype(int), np.clip(np.floor(ys).astype(int) + 1, 0, ph - 1)
    x0, x1 = np.floor(xs).astype(int), np.clip(np.floor(xs).astype(int) + 1, 0, pw - 1)
    dy = (ys - y0)[:, None]; dx = (xs - x0)[None, :]
    upsampled = ((1 - dy) * (1 - dx) * heat[np.ix_(y0, x0)]
                 + (1 - dy) * dx * heat[np.ix_(y0, x1)]
                 + dy * (1 - dx) * heat[np.ix_(y1, x0)]
                 + dy * dx * heat[np.ix_(y1, x1)])
    tau = float(np.quantile(upsampled, 1 - top_pct))
    return upsampled >= tau, upsampled


def _region_overlap(mask: np.ndarray, bbox: tuple[int, int, int, int],
                    W: int, H: int) -> dict[str, float]:
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2, y2 = min(W, x2), min(H, y2)
    gt = np.zeros_like(mask, dtype=bool)
    gt[y1:y2, x1:x2] = True
    inter = int((mask & gt).sum())
    union = int((mask | gt).sum())
    m_area, g_area = int(mask.sum()), int(gt.sum())
    return {
        "iou": inter / union if union else 0.0,
        "precision": inter / m_area if m_area else 0.0,
        "recall": inter / g_area if g_area else 0.0,
    }


# ------------------------------------------------------------------ #
# Per-sample eval
# ------------------------------------------------------------------ #
def eval_one(sample: dict, model) -> dict:
    entity = sample["expression"]
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": sample["image_path"]},
            {"type": "text",  "text": ATTN_PROMPT.format(entity=entity)},
        ],
    }]
    state = model.forward_with_attn(msgs)
    input_ids = state["input_ids"]
    attentions = state["attentions"]
    img_start, img_end = state["img_span"]
    Hp, Wp = state["patch_grid"]
    N_image = img_end - img_start

    tok = model.processor.tokenizer
    ent_positions = _find_entity_token_positions(
        input_ids, tok, entity, start_idx=img_end)
    if not ent_positions:
        return {"error": "no_entity_tokens"}

    layer_idxs = _mid_third_layers(len(attentions))
    per_token_maps = []
    for k in ent_positions:
        layer_maps = []
        for li in layer_idxs:
            a = attentions[li]                        # (1, H, T, T)
            if k >= a.shape[-2]:
                continue
            a_row = a[0, :, k, img_start:img_end]     # (H, N_image)
            a_row = a_row.mean(0)                     # head-mean
            layer_maps.append(a_row.float().cpu().numpy())
        if not layer_maps:
            continue
        stacked = np.stack(layer_maps, 0).mean(0)     # layer-mean
        per_token_maps.append(stacked.reshape(Hp, Wp))

    if not per_token_maps:
        return {"error": "no_valid_maps"}

    # Element-wise max across entity tokens (paper eq. 4)
    heat_2d = np.max(np.stack(per_token_maps, 0), 0)
    im = Image.open(sample["image_path"])
    W, H = im.size
    mask, _ = _heatmap_to_mask(heat_2d, (H, W))
    res = _region_overlap(mask, sample["bbox"], W, H)
    res["n_entity_tokens"] = len(per_token_maps)
    res["n_layers"] = len(layer_idxs)
    return res


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True,
                   help="Path to pathbind_grounding_500.jsonl (materialized)")
    p.add_argument("--model_module", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   help="sdpa or eager (FA2 does not expose attentions)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    print(f"[c1b] loading model from {args.model_module}", flush=True)
    mod = importlib.import_module(args.model_module)
    load = getattr(mod, "load_model")
    model = load(args.ckpt, gpus=args.gpus, attn_impl=args.attn_impl) \
        if args.ckpt else load(gpus=args.gpus, attn_impl=args.attn_impl)

    print(f"[c1b] loading data from {args.data}", flush=True)
    samples = [json.loads(l) for l in args.data.open()]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[c1b] {len(samples)} samples", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, s in enumerate(samples):
            t0 = time.time()
            try:
                res = eval_one(s, model)
            except Exception as e:
                res = {"error": type(e).__name__ + ": " + str(e)[:120]}
            res["sample_id"] = f"{s['split']}#{s['image_id']}#{s['bbox_id']}"
            res["split"] = s["split"]
            res["elapsed_sec"] = round(time.time() - t0, 2)
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i < 3 or (i + 1) % 50 == 0:
                iou = res.get("iou"); rec = res.get("recall")
                iou_s = f"{iou:.3f}" if iou is not None else "-"
                rec_s = f"{rec:.3f}" if rec is not None else "-"
                print(f"[{i+1:04d}/{len(samples)}] iou={iou_s} rec={rec_s}",
                      flush=True)

    # Aggregate
    rows = [json.loads(l) for l in args.out.open()]
    for split in ("testA", "testB"):
        valid = [r for r in rows if r.get("split") == split and "iou" in r]
        if not valid: continue
        iou = np.mean([r["iou"] for r in valid]) * 100
        pre = np.mean([r["precision"] for r in valid]) * 100
        rec = np.mean([r["recall"] for r in valid]) * 100
        print(f"[{split}]  n={len(valid)}  IoU={iou:.1f}  Pre={pre:.1f}  Rec={rec:.1f}")


if __name__ == "__main__":
    main()
