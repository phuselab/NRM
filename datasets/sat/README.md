# SAT — Speed-Accuracy Trade-off (SATBench)

16-way object recognition under a response deadline. Human observers answer at
one of several enforced deadlines ("beeps"), tracing out a speed-accuracy
trade-off curve per perturbation level.

## Contents

| File | Size | Description |
|---|---|---|
| `color-blur.txt` | 63 MB | Behavioural data, 45 subjects — colour→blur perturbation |
| `color-gray.txt` | 81 MB | Behavioural data, 58 subjects — colour→greyscale perturbation |
| `gray-noise.txt` | 63 MB | Behavioural data, 45 subjects — greyscale→noise perturbation |
| `data/` | 186 MB | Stimulus images: `0/`–`4/` (1,680 each) and `train/` (400) — rebuilt locally, see below |
| `generate_images.py` | — | rebuilds `data/` from the ImageNet validation set |
| `loader.py` | — | `SATDataset`, `sat_collate_fn`, `CATEGORIES` |

The three `.txt` files are SATBench's human data (OSF osf.io/2cpmb, CC-BY 4.0),
downloaded by `python fetch_assets.py sat-data`.

Each `.txt` is **newline-delimited JSON, one subject per line**, each line a
list of ~3,300 trial objects straight from the experiment runner (so it carries
consent/demographic rows and timing fields alongside the trials).
`SATDataset` flattens, filters and types it.

## The 16 categories

`airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog,
elephant, keyboard, knife, oven, truck` — order matters, it defines the label
indices. See `CATEGORIES` in `loader.py`.

## Loading

```python
from datasets.sat.loader import SATDataset, sat_collate_fn

ds = SATDataset(
    file_path="datasets/sat/color-blur.txt",
    image_dir="datasets/sat/data",
    load_images=True,
)
ds.compute_dinov2_embeddings(save_path="embeddings/sat/sat_color_blur_embeddings.pt")

ds.remove_rt_outliers(outlier_range_ms=[-100, 100])
ds.df = ds.df[ds.df["block"] != "cattut"].reset_index(drop=True)   # drop practice
train, val, test = ds.train_val_test_split(val_size=0.1, test_size=0.2,
                                           seed=42, shuffle=True)
```

Three steps that must not be skipped or reordered, since every reported number
assumes them:

1. `remove_rt_outliers(outlier_range_ms=[-100, 100])` — keeps responses within
   100 ms either side of the deadline.
2. Drop `block == "cattut"` — the practice/tutorial block.
3. Split with `seed=42`, 70/10/20.

If you filter rows after computing embeddings, filter the embeddings with the
same mask — `build_sat_data_dict` asserts the two agree rather than silently
misaligning.

## Stimulus images

The images are perturbed ILSVRC2012 validation images, which may not be
redistributed. `generate_images.py` rebuilds them with SATBench's pipeline from
your own copy of the ImageNet validation set (e.g. from the Kaggle *ImageNet
Object Localization Challenge*, 6.4 GB). Only 1,100 of its 50,000 images are
used (~140 MB): `imagenet_sources.csv` lists them, slot by slot, so the
selection does not have to be redone. Place the validation set as the Kaggle
archive lays it out, under `datasets/sat/ILSVRC/` (git-ignored); a directory
holding just the 1,100 listed files is enough:

```
datasets/sat/ILSVRC/
└── Data/CLS-LOC/val/        # ILSVRC2012_val_*.JPEG, flat
```

```bash
python fetch_assets.py sat-images
# or, with ImageNet elsewhere:
python fetch_assets.py sat-images --imagenet-val /path/to/ILSVRC/Data/CLS-LOC/val
```

Without a local copy, `--kaggle` downloads just the 1,100 listed images from
the Kaggle competition into `datasets/sat/ILSVRC/Data/CLS-LOC/val/` and then
builds the stimuli:

```bash
uv run --extra kaggle python fetch_assets.py sat-images --kaggle
```

It needs a Kaggle account and, once, the competition rules accepted on the
[rules page](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules)
(Kaggle offers no way to accept them from the command line; the script points
there if they are not). Without saved credentials it runs `kaggle auth login`,
an OAuth sign-in in your browser; on a machine without one, add
`--kaggle-no-browser` to open the link elsewhere and paste back the code.
An interrupted download resumes where it stopped.

`imagenet_sources.csv` was produced by SATBench's selection (16 categories,
sorted file lists, seed 42) over the full validation set and its
`LOC_val_solution.csv`; `generate_images.py --rebuild_sources` redoes it, and
needs both.

It writes 8,800 images (1,100 slots × 8 variants: blur 0/3/9, noise
0/0.04/0.16, colour/grey) and reports, for each behavioural file, how many of
the stimuli it references are present — all three must read 100 %. The noise
variants draw unseeded noise, so they are not bit-identical across rebuilds;
the others are. NRM does not need the images (it uses the shipped
embeddings); they are only needed to recompute the SAT embeddings or to train
the image-based SAT competitors.
