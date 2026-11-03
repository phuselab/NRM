# RDM embeddings — Motion energy (Gabor filter bank)

See [../README.md](../README.md) for the representation table, regeneration
commands and the alignment contract.

`rdm_embeddings.pt` is a dict with `embeddings` (58,438 × 150 × 24, float32)
and `video_indices` (58,438) as the alignment key. Fetched with
`python fetch_assets.py embeddings-rdm`.

Filter parameters (the released configuration): 8 directions, speeds
[0.5, 1.0, 2.0], λ = 8.0, σ_xy = 4.0, σ_t = 3.0, kernel 13 × 13 × 7.
Changing any of them makes `scripts/train_nrm_rdm.py` write to a
parameter-tagged filename instead, so the released cache is never overwritten
by an experiment.
