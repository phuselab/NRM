# MSDNet

We used code from the [original implementation](https://github.com/kalviny/MSDNet-PyTorch) and modified it for our experiments ([SAT-bench implementation](https://github.com/ajaysub110/satbench/tree/main/msdnet)).

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Training

Networks can be trained on the 16-class ImageNet dataset using the following command:

```bash
python train.py --tag trainMSDNet_MODE_COLOR --mode MODE --color COLOR --depth m --data imagenet --num_classes 16
```

where MODE is 'color', 'noise' or 'blur' and COLOR is 'color' or 'gray'.

## Testing

Trained networks can be tested on the same images used to test humans, using the following command:

```bash
python testonhuman.py --tag testMSDNet_MODE_COLOR --mode MODE --color COLOR --depth m --data imagenet --num_classes 16
--load_path LOAD_PATH
```

where MODE is 'color', 'noise' or 'blur', COLOR is 'color' or 'gray', and LOAD_PATH is the path to the saved model.

