# SC — Scene Categorisation

Binary natural-vs-manmade scene categorisation at 5 discriminability levels.
Stimuli are morphs between a natural and a manmade scene; the mixing proportion
sets how diagnostic the image is.

## Contents

| File | Size | Description |
|---|---|---|
| `experiment_1.csv` | 526 KB | Behavioural data: `subj_idx`, `img`, `cat`, `correct`, `rt`, `bad_trial`, `block`, `response` |
| `stimuli/` | 13 MB | 960 images in 10 folders (`manmade_0`…`manmade_80`, `natural_20`…`natural_100`), 96 each |
| `sun_filtered/` | 111 MB | SUN397 subset for RTify-SC / cRNN pre-training — rebuilt locally, see below |
| `build_sun_filtered.py` | — | rebuilds `sun_filtered/` from SUN397 |
| `loader.py` | — | `SCDataset`, `sc_collate_fn`, `project_embeddings` |

`experiment_1.csv` and `stimuli/` are the S1 File of Sofer, Crouzet & Serre
(2015, PLoS Comput Biol, CC-BY 4.0), downloaded by
`python fetch_assets.py sc-data`.

## Discriminability levels

The folder name encodes the morph: `natural_80` is 80 % natural, and
`manmade_20` is its complement. `loader.py` maps both to a single
*discriminability* in {0.2, 0.4, 0.6, 0.8, 1.0}:

```python
natural_N  -> N / 100
manmade_N  -> (100 - N) / 100
```

The `img` column is already stored relative to `stimuli/`
(`natural_80/sun_aoymjcfyfofogsle.jpg`), so `image_dir` points at `stimuli/`
itself — not at a subfolder.

## Loading

```python
from datasets.sc.loader import SCDataset, sc_collate_fn
import torch

ds = SCDataset(
    csv_path="datasets/sc/experiment_1.csv",
    image_dir="datasets/sc/stimuli",
    load_images=False,            # embeddings are pre-computed
)
train, val, test = ds.train_val_test_split(val_size=0.10, test_size=0.20, seed=42)

emb = torch.load("embeddings/sc/sc_embeddings.pt", weights_only=False)
train.embeddings, val.embeddings, test.embeddings = emb["train"], emb["val"], emb["test"]
```

Trials with `bad_trial == 1` are dropped on load.

The split is **by image**, not by trial, so no stimulus appears in two splits.
The 18-d projection is fitted on the training split only, which is why
`sc_embeddings.pt` stores the three splits together with the fitted `mean` and
`components`. Keep `seed=42` and the 0.10/0.20 proportions or rows and
embeddings will not correspond — the loaders assert on the lengths.

## SUN_filtered

8,762 images (7,900 train / 862 val, each split into `manmade/` and `natural/`),
used only to pre-train RTify-SC's CNN and the cRNN. They come from SUN397
(Xiao et al. 2010), whose images are not redistributed here;
`build_sun_filtered.py` rebuilds the subset from SUN397's copy on the Hugging
Face Hub (`1aurent/SUN397`, pinned revision, ~40 GB download): 7 man-made and
13 natural categories, every image of the experiment excluded, a stratified
7,900 / 862 split with seed 42, centre crop, 150×150 greyscale, mean luminance
128.

```bash
uv run --extra data python fetch_assets.py sun-filtered   # --extra data: pyarrow
```

It is not needed for NRM.
