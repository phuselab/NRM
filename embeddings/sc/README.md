# SC embeddings — DINOv2-S/14 + 18-d projection

See [../README.md](../README.md) for the representation table, regeneration
commands and the alignment contract.

`sc_embeddings.pt` is a dict with the already-projected splits `train`
(5105 × 18), `val` (727 × 18), `test` (1465 × 18), plus `n_components`, and the
fitted `mean` (384) and `components` (18 × 384) so the same projection can be
reapplied to new images.

The projection is a truncated SVD of the centred training features — a linear
dimensionality reduction to the 18 directions that carry the variance relevant
to scene categorisation. `project_embeddings()` in `datasets/sc/loader.py`
delegates to the implementation here, so there is one definition, not two.
