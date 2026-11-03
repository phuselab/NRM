# Competitor weights

Weights of the baselines NRM is compared against. **They are not distributed**:
the paper's competitor rows are reproduced from the prediction caches in
`outputs/` (`REFRESH = False` in the pipeline notebooks), and the models can be
retrained with their scripts. **None of these are needed to run, evaluate or
retrain NRM itself.**

| Directory | Model | Produced by |
|---|---|---|
| `rdm/` | RTify AlexNet+LSTM backbone, its cached states, accumulator head | `competitors/rtify/train_rdm.py` |
| `sat/` | CNet (parallel, serial), ConvRNN, MSDNet, SCAN × 3 conditions | `competitors/{cnet,convrnn,msdnet,scan}/train_sat.py` |
| `sc/` | RTify-SC (SUN-pre-trained CNN + 4 accumulator seeds), cRNN (hGRU + 4 RT-mapper seeds) | `competitors/rtify/train_sc.py`, `competitors/crnn/train_sc.py` |

Training keeps only the final weights, saved here with a `_REFRESH` suffix;
the pipeline notebooks evaluate them with `REFRESH_WEIGHTS = True`:

```
sat/cnet_parallel_<cond>_REFRESH.pt   td=1.0, all cascades active
sat/cnet_serial_<cond>_REFRESH.pt     td=0.0, sequential activation
sat/convrnn_<cond>_REFRESH.pt         BL-Net readout
sat/msdnet_<cond>_REFRESH.pt
sat/scan_<cond>_REFRESH.pt
rdm/rtify_backbone_REFRESH.pth        AlexNet+LSTM, final epoch
rdm/rtify_backbone_states_REFRESH.pth cached backbone states (~33 GB)
rdm/rtify_head_REFRESH.pth            accumulator head fitted to human RTs
sc/rtify_cnn_REFRESH.pth, sc/rtify_accumulator_seed{0-3}_REFRESH.pth
sc/crnn_mapper_seed{0-3}_REFRESH.pt   (and sc/crnn_hgru_REFRESH.pt if the hGRU is retrained)
```

`<cond>` ∈ {`color_blur`, `color_gray`, `gray_noise`}. CNet's parallel/serial
distinction is the `td` (temporal-difference) value, not a different
architecture.

## Third-party checkpoint

`sat/convrnn_backbone.h5` is the ImageNet-trained BL-Net released by its
authors (`bl_imagenet.h5`, Spoerer et al. 2020). It is read, never retrained,
and fetched from the source:

```bash
python fetch_assets.py convrnn-backbone
```
