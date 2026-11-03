# Competitor weights — SAT

CNet (parallel + serial), ConvRNN, MSDNet and SCAN for the three SAT
perturbation conditions are not distributed: `competitors/*/train_sat.py`
writes `<model>_<condition>_REFRESH.pt` here. The ConvRNN ImageNet backbone
(`convrnn_backbone.h5`) is fetched from its authors with
`python fetch_assets.py convrnn-backbone`.

See [../README.md](../README.md).
