# RDM — Random Dot Motion

Two-alternative motion-direction discrimination. 58,438 trials from 21
observers across 7 coherence levels.

## Contents

| File | Size | Description |
|---|---|---|
| `teddm_data.csv` | 2.5 MB | Raw behavioural data: reaction time, accuracy, subject ID, coherence |
| `rdm_nrm_dataset.pkl` | 7.1 MB | Pre-processed metadata DataFrame — trial table with split assignment and `video_index` |
| `loader.py` | — | `RDMDataset`, `load_rdm_nrm_dataset`, `build_rdm_nrm_dataset`, `rdm_collate_fn` |
| `motion_energy.py` | — | `MotionEnergyEmbedder` — Gabor spatiotemporal filter bank (Adelson & Bergen, 1985) |
| `rdm_generator.py` | — | Synthetic stimulus generator, deterministic per seed |

`teddm_data.csv` is the dot-motion data of Cochrane et al. (2023, *npj Science of
Learning*, doi:10.1038/s41539-023-00168-9), published on Zenodo
(doi:10.5281/zenodo.7025263, CC-BY 4.0) and downloaded by
`python fetch_assets.py rdm-data`. `rdm_nrm_dataset.pkl` is derived
from it (trial table, split assignment, video seeds).

## Loading

```python
from datasets.rdm.loader import load_rdm_nrm_dataset, PKL_OUT

ds = load_rdm_nrm_dataset(
    pkl_out=PKL_OUT,
    embeddings_path="embeddings/rdm/rdm_embeddings.pt",
)
ds.remove_rt_outliers_mad(mad_thresh=3.0, min_rt=0.05, max_rt=3.0)
train, val, test = ds.train_val_test_split(val_size=0.10, test_size=0.10, seed=42)
```

The split (80/10/10, `seed=42`) is stored in the pkl, and the shipped
embeddings and checkpoint assume it.

## The stimuli are not shipped

The random-dot movies are synthetic: 150 frames at 75 Hz, 126 px circular
aperture, 3 interleaved dot groups. Materialised they are ~130 GB, so the
repository ships the **generator plus the seeds** instead. Any trial is
reproducible exactly from its `video_index`:

```python
from datasets.rdm.rdm_generator import RDM_generator

movie = RDM_generator(direction=0, coherence=25.6, groups=3, size=126,
                      nDot=6, dotSize=3, speed=5, frames=150)   # (150, 126, 126)
```

Regenerate all of them only if you intend to recompute embeddings from pixels:

```python
ds.generate_and_save_videos()      # ~130 GB, hours
```

Needs OpenCV (a declared dependency, installed by `uv sync`).

## Stimulus parameters

Matching the original experiment: max duration 2000 ms, refresh 75 Hz
(150 frames), aperture 5.0° (125 px), dot speed 5.0°/s, dot size 0.1° (3 px),
density 16.7 dots/deg²/s, viewing distance 57 cm.
