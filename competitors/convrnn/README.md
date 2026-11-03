# ConvRNN

We adapted the code for the feedforward and recurrent neural network models proposed in the paper [Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision](https://doi.org/10.1371/journal.pcbi.1008215) and implemented in the SAT-bench ([SAT-bench implementation](https://github.com/ajaysub110/satbench/tree/main/convrnn)).

<br/><br/>

## Requirements

To install requirements:

```
pip install -r requirements.txt
```
<br/><br/>
## Training
### Data Set-up
This code expects you have access to the full ImageNet train data. Complete the following steps to subset to and train on the relevant 16 class dataset:
1. clone the [generalisation-humans-DNN](https://github.com/rgeirhos/generalisation-humans-DNNs) repository
2. `cd generalisation-humans-DNN/16-class-ImageNet`
3. `cp image_names EXPERIMENT_ROOT` where `EXPERIMENT_ROOT` is a directory where you would like to store the contents of iamge_names (text files with mappings to the appropriate subclasses for each of the 16 super classes)
4. Run `create_subset_image_directories.py` which will create directories of train and val images from the subset 16 class dataset. 
```
python create_subset_image_directories.py --dataset_root DATASET_ROOT --experiment_root EXPERIMENT_ROOT --new_dataset_root NEW_DATASET_ROOT --split_idxs_root SPLIT_IDX_ROOT
```
where
* `DATASET_ROOT` = Directory where you have downloaded or have access to the full ImageNet data. Should be the parent directory above train
* `EXPERIMENT_ROOT` = same as above
* `NEW_DATASET_ROOT` = Parent directory that will have two new directories: train and val each with subdirectories of images aligned to the 16 classes 
* `SPLIT_IDX_ROOT` = Directory where you choose to store train/val idx splits for reproducibility and consistency

### Pre-trained model download
This code relies on downloading pre-trained `bl_imagenet.h5`. Download instructions can be found [here](https://github.com/cjspoerer/rcnn-sat/blob/master/restore_and_extract_activations.ipynb) 

_, msg = urllib.request.urlretrieve(
    'https://osf.io/9td5p/download', 'bl_imagenet.h5')
### Run training scripts
This repository supports training on a 16 class subset of the ImageNet2012 dataset under 4 different image perturbation regimes:
* color (default - no perturbations applied)
* grayscale 
* gaussian noise applied to grayscale images
* gaussian blur filter applied to color images


<br/><br/>

## Evaluation 
### Test Data Set-up
The convRNN dataloader expects image data in a format that is slightly different from the other models in this repsoitory. Therefore, in order to evaluate the trained convRNN model on the data collected for human experiments found at [https://osf.io/2cpmb/] you must run the following:
```
python reformat_test_data_dir.py --old_dir OLD_DIR --new_dir NEW_DIR
```
where 
* `OLD_DIR` = directory where you have downloaded data from human experiments from [https://osf.io/2cpmb/]
* `NEW_DIR` = new directory where you will store reformated test data for convRNN model

