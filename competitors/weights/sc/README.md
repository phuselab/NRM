# Competitor weights — SC

RTify-SC (SUN-pre-trained CNN + 4 accumulator seeds) and cRNN (hGRU + 4
RT-mapper seeds) are not distributed: `competitors/rtify/train_sc.py` and
`competitors/crnn/train_sc.py` write their `_REFRESH` weights here. Their
pre-training needs `datasets/sc/sun_filtered/`
(`python fetch_assets.py sun-filtered`).

See [../README.md](../README.md).
