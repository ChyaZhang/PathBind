"""Rebuild PathBind-VQA locally from the shipped sample-id manifest.

PathBind-VQA does NOT redistribute question/option/image content of its five
public sources (PathMMU, Path-VQA, Quilt-VQA, MedXpert-Path, OmniMed-Bright).
You must first download those benchmarks yourself, then run this script with
each benchmark's local directory to materialize the 1{,}500-sample TSV.

The manifest (pathbind_vqa_1500_manifest.tsv) contains only:
    index,dimension,source_dataset,source_sample_id,answer_type


Usage:
    python build.py \\
        --pathmmu_dir  <local PathMMU images+questions>  \\
        --pathvqa_dir  <local Path-VQA>  \\
        --quiltvqa_dir <local Quilt-VQA>  \\
        --omnimed_dir  <local OmniMed-Bright>  \\
        --medxpert_dir <local MedXpert-Path>  \\
        --out          pathbind_vqa_1500.tsv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).with_name("pathbind_vqa_1500_manifest.tsv")

SOURCE_LOADERS = {
    # Each loader must return a dict:
    #   sample_id -> {"question": str, "options": [str, ...], "answer": str,
    #                 "answer_type": "mcq"|"yn", "image_path": str}
    # Fill this in per source. The stubs below assume the local layout used by
    # our reference reimplementation; adjust to your local copy as needed.
}


def load_pathmmu(root: Path) -> dict:
    """PathMMU test-tiny + test-all. Expects `samples.jsonl` and `images/`."""
    samples_path = root / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    out = {}
    for line in samples_path.open():
        s = json.loads(line)
        out[s["sample_id"]] = {
            "question": s["question"],
            "options": s["options"],
            "answer": s["answer"],
            "answer_type": "mcq",
            "image_path": str(root / "images" / f"{s['sample_id']}.jpg"),
        }
    return out


def load_pathvqa(root: Path) -> dict:
    """Path-VQA closed. Uses only yes/no subset."""
    return load_pathmmu(root)   # same layout


def load_quiltvqa(root: Path) -> dict:
    return load_pathmmu(root)


def load_omnimed(root: Path) -> dict:
    return load_pathmmu(root)


def load_medxpert(root: Path) -> dict:
    return load_pathmmu(root)


LOADER_REGISTRY = {
    "pathmmu": load_pathmmu,
    "pathvqa": load_pathvqa,
    "quiltvqa": load_quiltvqa,
    "omnimed": load_omnimed,
    "medxpert": load_medxpert,
}


def main():
    p = argparse.ArgumentParser()
    for k in LOADER_REGISTRY:
        p.add_argument(f"--{k}_dir", type=Path, required=True,
                       help=f"Local root of the {k} benchmark")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).with_name("pathbind_vqa_1500.tsv"))
    args = p.parse_args()

    print(f"[pathbind-vqa] loading 5 source benchmarks ...", file=sys.stderr)
    sources = {k: fn(getattr(args, f"{k}_dir"))
               for k, fn in LOADER_REGISTRY.items()}
    for k, d in sources.items():
        print(f"    {k}: {len(d)} samples", file=sys.stderr)

    print(f"[pathbind-vqa] reading manifest {MANIFEST} ...", file=sys.stderr)
    manifest_rows = list(csv.DictReader(MANIFEST.open(), delimiter="\t"))
    print(f"    manifest rows: {len(manifest_rows)}", file=sys.stderr)

    hdr = ["index", "question"] + list("ABCDEFGHIJKLMNOP") + [
        "image_path", "answer", "answer_type", "source_dataset", "dimension"
    ]
    out_rows = []
    missing = []
    for m in manifest_rows:
        ds = m["source_dataset"]
        sid = m["source_sample_id"]
        rec = sources.get(ds, {}).get(sid)
        if rec is None:
            missing.append((ds, sid))
            continue
        opts = rec["options"] + [""] * (16 - len(rec["options"]))
        out_rows.append([
            m["index"], rec["question"], *opts,
            rec["image_path"], rec["answer"], rec["answer_type"],
            ds, m["dimension"],
        ])

    print(f"[pathbind-vqa] materialized: {len(out_rows)}, missing: {len(missing)}",
          file=sys.stderr)
    if missing:
        print(f"    first 10 missing: {missing[:10]}", file=sys.stderr)

    with args.out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(hdr)
        w.writerows(out_rows)
    print(f"[pathbind-vqa] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
