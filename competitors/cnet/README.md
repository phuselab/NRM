# Cascaded Networks

We used code from [Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss](https://arxiv.org/abs/2102.09808) ([code](https://github.com/michael-iuzzolino/CascadedNets?tab=MIT-1-ov-file)) and modified it for our experiments ([SAT-bench implementation](https://github.com/ajaysub110/satbench/tree/main/cnet)). 

## Requirements

To install requirements:

```
pip install -r requirements.txt
```
<br/><br/>
## CascadedNet-Specific Data Set Up
This code expects access to the full ImageNet/train dataset and then subsequently subsets to the 16-class subset. Necessary steps for doing this are:
1. clone the [generalisation-humans-DNN](https://github.com/rgeirhos/generalisation-humans-DNNs) repository
2. `cd generalisation-humans-DNN/16-class-ImageNet`
3. `cp image_names EXPERIMENT_ROOT` where `EXPERIMENT_ROOT` is the parent output directory you've assigned for storing model outputs

<br/><br/>

## Training
This repository supports training on a 16 class subset of the ImageNet2012 dataset initially developed here [https://github.com/rgeirhos/generalisation-humans-DNNs] under 4 different image perturbation regimes:
* color (default - no perturbations applied)
* grayscale 
* gaussian noise applied to grayscale images
* gaussian blur filter applied to color images


Use `train.sh` as a template to set your own hyperparameters, which launches `train.py`.
Below are shell commands you can run to replicate training for the above perturbations where:
* `DATASET_ROOT` = parent directory for imagenet data. Should have at least subdirectory train
* `EXPERIMENT_ROOT` = parent output directory for storing model outputs and results (subdirectories for different types of experiments are generated automatically during run time)
* `SPLIT_IDX_ROOT` = directory for storing train/val idx splits for reproducibility and consistency
* `HUMAN_DATA_DIR` = directory where you have downloaded data from human experiments

Note that the arguments passed in the following shell commands specify key directories, the type of CNet model to run (`parallel` or `serial`), temporal difference hyperparameter (`LAMBDA`) and the image perturbation regime. The current hyperparameters in train.sh are configured to reproduce results in this paper but can be adjusted as needed or desired. Additionally, we include examples for reproducing results with the parallel and serial CNet architectures respectively.

### Color
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 false false false
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 false false false
```

### Grayscale
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 true false false
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 true false false
```

### Gaussian Noise on Grayscale Images
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 true true false
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 true true false
```

### Gaussian Blur on Color Images
```
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR parallel 1.0 false false true
bash train.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR serial 0.0 false false true
```

<br/><br/>
## Evaluation
Use `eval.sh` to run `eval.py`, which loads and evaluates the model(s) stored in `EXPERIMENT_ROOT` + `EXPERIMENT_NAME`. Unlike train, it is not necessary to specify the image perturbation regmine: `eval.py` will automatically run on all model outputs located under `EXPERIMENT_ROOT` + `EXPERIMENT_NAME`. Not that if `--keep_logits` is specified, generate and store the logits for all examples in the specified dataset. 

Similar to `train.sh`, we provide an example shell command for running `eval.sh` where `DATASET_ROOT`, `EXPERIMENT_ROOT`, `SPLIT_IDX_ROOT`, and `HUMAN_DATA_DIR` are defined as above. Here we also specify `DATASET_KEY` which specifies which testing dataset to evaluate on: `test` (standard ImageNet test dataset) or `test_human` (human test data for this project)

```
bash eval.sh DATASET_ROOT EXPERIMENT_ROOT SPLIT_IDX_ROOT HUMAN_DATA_DIR DATASET_KEY
```
### Outputs
The CNet architecture automatically evaluates all images at every timestep. Therefore outputs are saved as a dictionary with the following keys: `logits`, `softmax`, `predictions`, `correct`, `target`, `target_confidence`, `prediction_confidence` where all dim 0 of all values = 9 given there are 9 timesteps. 

### Human Data-specific Analysis
For appropriate comparisons to human performance, we only evaluate certain images at certain timesteps. Therefore to replicate performance results depicted in the paper and obtain a dataframe of predictions/ targets aligned with the relevant timesteps, run `split_exit_accuracies.py`. `DATASET_ROOT` and `HUMAN_DATA_DIR` are the same as specified above.

```
python split_exit_accuracies.py --experiments_dir EXPERIMENTS_ROOT --human_data_dir HUMAN_DATA_DIR --output_dir OUTPUT_DIR --no-get_acc_by_cat
```
For obtaining accuracies by categories:
```
python split_exit_accuracies.py --experiments_dir EXPERIMENTS_ROOT --human_data_dir HUMAN_DATA_DIR --output_dir OUTPUT_DIR --no-get_acc_by_cat
```


