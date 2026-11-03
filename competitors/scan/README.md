# SCAN

SCAN (Zhang et al. 2019, [pytorch-scalable-neural-networks](https://github.com/ArchipLab-LinfengZhang/pytorch-scalable-neural-networks)),
in the version adapted for the speed–accuracy benchmark by
[SATBench](https://github.com/ajaysub110/satbench) (Subramanian et al. 2022,
folder `scan/`).

Neither repository publishes a licence for this code, so it is **not
redistributed here**. `train_sat.py` is ours; the model code is downloaded
from SATBench, pinned to commit `847c57811a67c2eb2f8424a1974f6b4078bb340f` and
checked against the files we used:

```bash
python fetch_assets.py scan-code
```

This places `sresnet.py`, `small_resnet.py`, `train.py`, `test.py`,
`utils.py`, `custom_transforms.py` and `requirements.txt` in this directory.
Then train and evaluate SCAN on a SAT condition with

```bash
python competitors/scan/train_sat.py --perturb color-blur
```
