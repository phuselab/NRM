# Stimulus embeddings

NRM is conditioned on a stimulus embedding `c`; drift, diffusion and threshold
are all functions of it. Each benchmark uses the representation appropriate to
its stimulus, and each directory holds both the cached tensor and the script
that produced it.

| Benchmark | Representation | Dim | File | Size |
|---|---|---|---|---|
| RDM | Motion energy — Gabor bank, 8 directions × 3 speeds, **per frame** | 150 × 24 | `rdm/rdm_embeddings.pt` | 803 MB (GitHub Release) |
| SAT | DINOv2-S/14 patch-token mean, per stimulus | 384 | `sat/sat_<perturbation>_embeddings.pt` | 76–98 MB each (GitHub Release) |
| SC | DINOv2-S/14 then linear projection fitted on the train split | 18 | `sc/sc_embeddings.pt` | 0.4 MB (in git) |

RDM is the odd one out: its embedding is a **time series**, one 24-d vector per
frame, which is why RDM uses `TimeEmbeddedNRM` — the conditioning evolves
during the trial, so drift, diffusion and threshold are all time-varying. SAT
and SC use a single static vector per stimulus.

## Regenerating

The shipped tensors were produced by exactly these commands:

```bash
python embeddings/rdm/compute_embeddings.py                              # needs the videos
python embeddings/sat/compute_embeddings.py --perturbation color-blur    # minutes on a GPU
python embeddings/sc/compute_embeddings.py                               # minutes on a GPU
```

RDM is the expensive one: motion energy is computed from pixels, so it needs
the ~130 GB video corpus materialised first (`datasets/rdm/README.md`). SAT and
SC embed shipped images and are cheap to redo.

`SATDataset.compute_dinov2_embeddings(save_path=...)` writes to the same path
and is called by `pipelines/sat.ipynb`, so the SAT files appear on first run if
absent.

## Alignment

Every cache is positionally aligned to a dataset split, and a mismatch is a
silent correctness bug, so the loaders check rather than trust:

- **RDM** stores `video_indices` alongside `embeddings`; `load_rdm_nrm_dataset`
  reorders rows onto the dataframe's `video_index`.
- **SAT** stores `image_paths`; `compute_dinov2_embeddings` realigns by path
  when the row count differs, and `build_sat_data_dict` raises if lengths
  disagree.
- **SC** stores the `train`/`val`/`test` splits separately, plus the fitted
  `mean` and `components`. The projection is fitted on **train only**, so the
  split (`seed=42`, 0.10/0.20) must match the one used at fit time.
