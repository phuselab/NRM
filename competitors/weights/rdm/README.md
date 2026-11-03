# Competitor weights — RDM

RTify's AlexNet+LSTM backbone, its cached per-frame states (~33 GB) and its
accumulator head are not distributed. They are **not needed to reproduce the
paper**: `pipelines/rdm.ipynb` reads RTify's per-trial predictions from
`outputs/rdm/rtify_predictions.npz`.

`competitors/rtify/train_rdm.py` trains the backbone on synthetic RDM movies,
caches its states and fits the head, writing `rtify_backbone_REFRESH.pth`,
`rtify_backbone_states_REFRESH.pth` and `rtify_head_REFRESH.pth` here;
`pipelines/rdm.ipynb` evaluates them with `REFRESH_WEIGHTS = True`. The draw is
seeded; see `RTifyModel.sample_onto_trials` for the two draws it makes per
coherence and why their order matters.

See [../README.md](../README.md).
