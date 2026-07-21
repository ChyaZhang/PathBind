"""Rebuild PathBind-Grounding locally from the shipped sample-id manifest.

The 500 expert-approved (image, bbox, expression) triplets are drawn from
PathVG (Zhong et al., 2025) testA / testB splits. To respect the source
license we ship only sample IDs; you supply the images from your local
PathVG copy.

Usage:
    python build.py --pathvg_dir <local PathVG> --out pathbind_grounding_500.jsonl
"""
import argparse
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("pathbind_grounding_500_manifest.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pathvg_dir", type=Path, required=True,
                   help="Root of local PathVG (must contain testA/ and testB/ image dirs)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).with_name("pathbind_grounding_500.jsonl"))
    args = p.parse_args()

    with args.out.open("w") as out_f:
        n_ok, n_miss = 0, 0
        for line in MANIFEST.open():
            r = json.loads(line)
            img_path = args.pathvg_dir / r["split"] / r["image_id"]
            if not img_path.exists():
                n_miss += 1
                continue
            r["image_path"] = str(img_path)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[pathbind-grounding] materialized: {n_ok}, missing: {n_miss}")
    print(f"[pathbind-grounding] wrote {args.out}")


if __name__ == "__main__":
    main()
