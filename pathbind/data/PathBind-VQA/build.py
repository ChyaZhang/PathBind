"""Rebuild PathBind-VQA locally from the upstream releases + the shipped crosswalk.

PathBind-VQA does NOT redistribute question/option/image content of its five
public sources. Download them yourself (revisions pinned in the crosswalk),
then run this script to materialize the 1,500-sample TSV.

pathbind_vqa_1500_crosswalk.tsv maps every manifest row to one upstream QA
record (row order == manifest order):

    row, index, dimension, source_dataset, answer_type,
    upstream_repo, upstream_revision, upstream_split,
    upstream_qa_id, upstream_image, question_sha1

  pathmmu   jamessyx/PathMMU            data.json[subset]["test"][No]
  pathvqa   flaviagiammarino/path-vqa   0-based row of the `test` split
                                        (shards concatenated in file order,
                                        before any yes/no filtering)
  quiltvqa  wisdomik/Quilt_VQA          quiltvqa_test_w_ans.json, record `id`
  medxpert  TsinghuaC3I/MedXpertQA      MM/test.jsonl, record `id`
  omnimed   OmniMedVQA (Restricted)     "BRIGHT Challenge.json", question_id
                                        BRIGHT Challenge_NNNN (= list position)

question_sha1 = sha1(lower(whitespace-normalized question))[:12]; every row is
checked against it, so a wrong match fails loudly instead of silently.

Expected local layouts (images may sit anywhere below each dir; they are
looked up by file name):
    --pathmmu_dir   data.json + unzipped images.zip
    --pathvqa_dir   the HF repo snapshot (data/test-*.parquet); needs pandas
    --quiltvqa_dir  quiltvqa_test_w_ans.json + unzipped quilt_vqa.zip
    --medxpert_dir  MM/test.jsonl + unzipped images.zip
    --omnimed_dir   OmniMedVQA root (QA_information/.../BRIGHT Challenge.json
                    + Images/)

Every source dir is optional; rows of a missing source (or with a missing
image) are skipped and reported.

Usage:
    python build.py --pathmmu_dir ... --pathvqa_dir ... --quiltvqa_dir ... \\
        --medxpert_dir ... --omnimed_dir ... --out pathbind_vqa_1500.tsv
"""
import argparse
import csv
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CROSSWALK = Path(__file__).with_name("pathbind_vqa_1500_crosswalk.tsv")
LETTERS = "ABCDEFGHIJKLMNOP"


def qhash(q: str) -> str:
    return hashlib.sha1(" ".join(q.split()).lower().encode()).hexdigest()[:12]


def image_index(root: Path) -> dict:
    """file name and stem -> path, for every file below root."""
    idx = {}
    for p in root.rglob("*"):
        if p.is_file():
            idx.setdefault(p.name, p)
            idx.setdefault(p.stem, p)
    return idx


# Each loader returns fetch(crosswalk_row) -> (question, options, answer, image_path)
# and raises KeyError when the image is not available locally.

def load_pathmmu(root: Path, out_img: Path):
    data = json.load((root / "data.json").open())
    imgs = image_index(root)

    def fetch(c):
        subset, split = c["upstream_split"].split("/")
        r = data[subset][split][int(c["upstream_qa_id"])]
        opts = [o[3:] if o[1:3] == ") " else o for o in r["options"]]
        return r["question"], opts, r["answer"][0], str(imgs[r["img"]])
    return fetch


def load_pathvqa(root: Path, out_img: Path):
    import pandas as pd
    files = sorted(glob.glob(str(root / "data" / "test-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def fetch(c):
        n = int(c["upstream_qa_id"])
        p = out_img / f"pathvqa_test_{n}.jpg"
        if not p.exists():
            p.write_bytes(df.image[n]["bytes"])
        return df.question[n], [], df.answer[n].strip().lower(), str(p)
    return fetch


def load_quiltvqa(root: Path, out_img: Path):
    data = {r["id"]: r for r in json.load((root / "quiltvqa_test_w_ans.json").open())}
    imgs = image_index(root)

    def fetch(c):
        r = data[int(c["upstream_qa_id"])]
        return r["question"], [], r["yes_no_answer"].strip().lower(), str(imgs[r["image"]])
    return fetch


def load_medxpert(root: Path, out_img: Path):
    data = {r["id"]: r for r in map(json.loads, (root / "MM" / "test.jsonl").open())}
    imgs = image_index(root)

    def fetch(c):
        r = data[c["upstream_qa_id"]]
        q = r["question"].split("\nAnswer Choices:")[0]   # options are listed separately
        return q, list(r["options"].values()), r["label"], str(imgs[c["upstream_image"]])
    return fetch


def load_omnimed(root: Path, out_img: Path):
    (qa,) = root.rglob("BRIGHT Challenge.json")
    data = json.load(qa.open())
    imgs = image_index(root)

    def fetch(c):
        r = data[int(c["upstream_qa_id"].rsplit("_", 1)[1])]
        opts = [r[f"option_{L}"] for L in "ABCD" if r.get(f"option_{L}")]
        ans = LETTERS[opts.index(r["gt_answer"])]
        return r["question"], opts, ans, str(imgs[c["upstream_image"]])
    return fetch


LOADER_REGISTRY = {
    "pathmmu": load_pathmmu,
    "pathvqa": load_pathvqa,
    "quiltvqa": load_quiltvqa,
    "medxpert": load_medxpert,
    "omnimed": load_omnimed,
}


def main():
    p = argparse.ArgumentParser()
    for k in LOADER_REGISTRY:
        p.add_argument(f"--{k}_dir", type=Path, help=f"Local root of the {k} release")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).with_name("pathbind_vqa_1500.tsv"))
    args = p.parse_args()
    out_img = args.out.with_suffix(".images")
    out_img.mkdir(parents=True, exist_ok=True)

    fetchers = {}
    for k, fn in LOADER_REGISTRY.items():
        d = getattr(args, f"{k}_dir")
        if d is not None:
            print(f"[pathbind-vqa] loading {k} from {d}", file=sys.stderr)
            fetchers[k] = fn(d, out_img)

    crosswalk = list(csv.DictReader(CROSSWALK.open(), delimiter="\t"))
    hdr = ["index", "question", *LETTERS, "image_path", "answer",
           "answer_type", "source_dataset", "dimension"]
    out_rows, missing = [], Counter()
    for c in crosswalk:
        ds = c["source_dataset"]
        if ds not in fetchers:
            missing[(ds, "source dir not given")] += 1
            continue
        try:
            q, opts, ans, img = fetchers[ds](c)
        except KeyError:
            missing[(ds, "image not found")] += 1
            continue
        if qhash(q) != c["question_sha1"]:
            sys.exit(f"question_sha1 mismatch at row {c['row']} ({c['index']}): "
                     f"wrong upstream revision?")
        # newlines -> spaces, as in the TSVs we evaluated on
        q, *opts = [t.replace("\n", " ").strip() for t in (q, *opts)]
        out_rows.append([c["index"], q, *opts, *[""] * (16 - len(opts)),
                         img, ans, c["answer_type"], ds, c["dimension"]])

    print(f"[pathbind-vqa] materialized {len(out_rows)}/{len(crosswalk)}", file=sys.stderr)
    for (ds, why), n in sorted(missing.items()):
        print(f"    skipped {n:4d} {ds}: {why}", file=sys.stderr)

    with args.out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(hdr)
        w.writerows(out_rows)
    print(f"[pathbind-vqa] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
