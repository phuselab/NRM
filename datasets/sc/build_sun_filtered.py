#!/usr/bin/env python3
"""Rebuild the SUN397 pre-training subset for RTify-SC and cRNN (Xiao et al. 2010)."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent

HF_REPO = "1aurent/SUN397"
HF_REVISION = "07ddcfe3001fcab521ad88230f2064dc335237e7"

TRAIN_SIZE = 7900
VAL_SIZE = 862
RANDOM_SEED = 42
JPEG_QUALITY = 95
TARGET_SIZE = 150

MANMADE_LABELS = {
    "/s/skyscraper",
    "/h/highway",
    "/s/street",
    "/t/tower",
    "/a/alley",
    "/a/apartment_building/outdoor",
    "/a/amphitheater",
}

NATURAL_LABELS = {
    "/b/beach",
    "/d/desert/sand",
    "/d/desert/vegetation",
    "/f/field/cultivated",
    "/u/underwater/coral_reef",
    "/i/iceberg",
    # forest (broadleaf + needleleaf + rainforest)
    "/f/forest/broadleaf",
    "/f/forest/needleleaf",
    "/r/rainforest",
    # mountain group (mountain + mountain_snowy + coast + cliff)
    "/m/mountain",
    "/m/mountain_snowy",
    "/c/coast",
    "/c/cliff",
}


def build_exclusion_set(csv_path: Path) -> set:
    stems = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stems.add(Path(row["img"].strip()).stem)
    return stems


def preprocess(img: Image.Image) -> Image.Image:
    # 1. Centre-crop to square
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    # 2. Resize
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    # 3. Greyscale
    img = img.convert("L")
    # 4. Global luminance normalisation (mean → 128)
    arr = np.array(img, dtype=np.float32)
    mean_val = arr.mean()
    if mean_val > 0:
        arr *= 128.0 / mean_val
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def label_names(readme_path: Path) -> list[str]:
    """Class names in label-index order, from the dataset card's features."""
    names = {}
    for m in re.finditer(r"^\s+'(\d+)':\s+(\S+)\s*$", readme_path.read_text(), re.M):
        names[int(m.group(1))] = m.group(2)
    return [names[i] for i in range(len(names))]


def hub_files() -> tuple[list[Path], Path]:
    """Download (or reuse from the HF cache) the pinned parquet shards and card."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(
        HF_REPO, repo_type="dataset", revision=HF_REVISION)
        if f.startswith("data/train-") and f.endswith(".parquet"))
    shards = [Path(hf_hub_download(HF_REPO, f, repo_type="dataset",
                                   revision=HF_REVISION)) for f in files]
    readme = Path(hf_hub_download(HF_REPO, "README.md", repo_type="dataset",
                                  revision=HF_REVISION))
    return shards, readme


def build(shards: list[Path], readme: Path, exp1_csv: Path, output_root: Path):
    import pyarrow.parquet as pq

    names = label_names(readme)
    label_to_binary = {}
    for i, name in enumerate(names):
        if name in MANMADE_LABELS:
            label_to_binary[i] = "manmade"
        elif name in NATURAL_LABELS:
            label_to_binary[i] = "natural"

    # ── Index: label and filename stem of every row, shard by shard ──────────
    labels_col, all_stems, row_loc = [], [], []
    for si, shard in enumerate(shards):
        t = pq.read_table(shard, columns=["image", "label"])
        labels = t.column("label").to_pylist()
        paths = t.column("image").combine_chunks().field("path").to_pylist()
        for ri, (lbl, p) in enumerate(zip(labels, paths)):
            labels_col.append(lbl)
            all_stems.append(Path(p).stem if p else None)
            row_loc.append((si, ri))
    print(f"  SUN397: {len(labels_col):,} images, {len(names)} categories")

    # ── Step 1: filter by category ───────────────────────────────────────────
    kept_indices = [i for i, lbl in enumerate(labels_col) if lbl in label_to_binary]
    print(f"  After category filter: {len(kept_indices):,}")

    # ── Step 2: exclude experiment-1 images ──────────────────────────────────
    exclusion_set = build_exclusion_set(exp1_csv)
    filtered_indices = [i for i in kept_indices
                        if not (all_stems[i] and all_stems[i] in exclusion_set)]
    print(f"  After excluding experiment_1 images: {len(filtered_indices):,}")

    # ── Step 3: stratified split ─────────────────────────────────────────────
    required = TRAIN_SIZE + VAL_SIZE
    if len(filtered_indices) < required:
        raise RuntimeError(f"Not enough images: have {len(filtered_indices):,}, "
                           f"need {required:,}.")
    binary_labels = [label_to_binary[labels_col[i]] for i in filtered_indices]
    train_indices, val_indices = train_test_split(
        filtered_indices, train_size=TRAIN_SIZE, test_size=VAL_SIZE,
        stratify=binary_labels, random_state=RANDOM_SEED,
    )

    # ── Step 4: preprocess and save ──────────────────────────────────────────
    for split in ("train", "val"):
        for cls in ("natural", "manmade"):
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)

    wanted = {}
    for split, indices in (("train", train_indices), ("val", val_indices)):
        for idx in indices:
            wanted[idx] = split

    # Filenames are assigned in split-list order, exactly as the stems are seen.
    out_name = {}
    seen_stems = {"train": set(), "val": set()}
    for split, indices in (("train", train_indices), ("val", val_indices)):
        for idx in indices:
            sun_label = names[labels_col[idx]]
            stem = all_stems[idx] or f"sun_{sun_label.replace('/', '_')}_{idx}"
            filename = f"{stem}.jpg"
            if stem in seen_stems[split]:
                subcat = sun_label.strip("/").replace("/", "_")
                filename = f"{stem}_{subcat}.jpg"
            seen_stems[split].add(stem)
            out_name[idx] = output_root / split / label_to_binary[labels_col[idx]] / filename

    by_shard = {}
    for idx in wanted:
        si, ri = row_loc[idx]
        by_shard.setdefault(si, []).append((ri, idx))
    for si, rows in sorted(by_shard.items()):
        t = pq.read_table(shards[si], columns=["image"])
        images = t.column("image").combine_chunks().field("bytes")
        for ri, idx in rows:
            # Decoded as the Hugging Face `datasets` Image feature does:
            # EXIF orientation applied, then RGB.
            img = ImageOps.exif_transpose(
                Image.open(io.BytesIO(images[ri].as_py()))).convert("RGB")
            preprocess(img).save(out_name[idx], format="JPEG", quality=JPEG_QUALITY)
    print(f"  Saved {len(train_indices):,} train + {len(val_indices):,} val images "
          f"to {output_root}")


def main():
    ap = argparse.ArgumentParser(description="Rebuild datasets/sc/sun_filtered/ "
                                             "from SUN397 on the Hugging Face Hub.")
    ap.add_argument("--output_dir", default=str(_HERE / "sun_filtered"))
    ap.add_argument("--exp1_csv", default=str(_HERE / "experiment_1.csv"))
    args = ap.parse_args()
    try:
        import pyarrow  # noqa: F401  (fail before the ~40 GB download, not after)
    except ImportError:
        sys.exit("pyarrow is required: run `uv sync --extra data` first.")
    shards, readme = hub_files()
    build(shards, readme, Path(args.exp1_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
