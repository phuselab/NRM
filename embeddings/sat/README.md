# SAT embeddings — DINOv2-S/14 features

See [../README.md](../README.md) for the representation table, regeneration
commands and the alignment contract.

One file per perturbation: `sat_color_blur_embeddings.pt`,
`sat_color_gray_embeddings.pt`, `sat_gray_noise_embeddings.pt` (note the
underscores — the behavioural files use hyphens). Each is a dict with
`embeddings` (N × 384), `dim`, and `image_paths` for alignment.
