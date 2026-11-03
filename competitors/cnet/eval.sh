#!/bin/bash

DATASET_ROOT=$1   #"/scratch/work/public/imagenet"  # Specify location of datasets
EXPERIMENT_ROOT=$2  # Specify experiment root
SPLIT_IDXS_ROOT=$3 # Specify root of dataset split_idxs
TEST_DATASET_ROOT=$4 

# Experiment name to evaluate
MODEL="resnet18"  # resnet18, resnet34, resnet50, densenet_cifar
DATASET_NAME="ImageNet2012_16classes_rebalanced" #"ImageNet2012_16classes_rebalanced"  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
EXPERIMENT_NAME="${MODEL}_${DATASET_NAME}"

TRAIN_MODE="cascaded"  # baseline, cascaded_seq, cascaded
CASCADED_SCHEME= "parallel" #regardless of what is set here, eval.py will evaluate all results stored under os.path.join(EXPERIMENT_ROOT, EXPERIMENT_NAME)
DATASET_KEY=$5  # options are test or test_human
BATCH_SIZE=128

TDL_MODE="OSD"  # OSD, EWS, noise
TDL_ALPHA=0.9
NOISE_VAR=0.0  # Used for noise kernel only
N_TIMESTEPS=70  # Used for EWS kernel only

#Image perturbations
GRAYSCALE=false
GAUSS_NOISE=false
GAUSS_NOISE_STD=0.0
EVAL_GAUSS_NOISE_RANGE=(0.09 0.11 0.13 0.15 0.17 0.19 0.21 0.23 0.25 0.27) #can update with whatever noise range is desired to test. this is only applicable on non human test dataset
BLUR=false
BLUR_STD=0.0
EVAL_BLUR_RANGE=(0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5) #can update with whatever noise range is desired to test. this is only applicable on non human test dataset

DEVICE=0
KEEP_LOGITS=true
KEEP_EMBEDDINGS=false
FORCE_OVERWRITE=true
DEBUG=false

cmd=( python ../CascadedNets/eval.py )   # create array with one element
cmd+=( --device $DEVICE )
cmd+=( --dataset_root $DATASET_ROOT )
cmd+=( --test_dataset_root $TEST_DATASET_ROOT )
cmd+=( --dataset_name $DATASET_NAME )
cmd+=( --dataset_key $DATASET_KEY )
cmd+=( --split_idxs_root $SPLIT_IDXS_ROOT )
cmd+=( --experiment_root $EXPERIMENT_ROOT )
cmd+=( --experiment_name $EXPERIMENT_NAME )
cmd+=( --train_mode $TRAIN_MODE )
cmd+=( --batch_size $BATCH_SIZE )
cmd+=( --cascaded_scheme $CASCADED_SCHEME )
cmd+=( --tdl_mode $TDL_MODE )
cmd+=( --tdl_alpha $TDL_ALPHA )
cmd+=( --noise_var $NOISE_VAR )
cmd+=( --n_timesteps $N_TIMESTEPS )
cmd+=( --gauss_noise_std $GAUSS_NOISE_STD )
cmd+=( --eval_blur_range "${EVAL_GAUSS_NOISE_RANGE[@]}" )
cmd+=( --blur_std $BLUR_STD )
cmd+=( --eval_blur_range "${EVAL_BLUR_RANGE[@]}" )

${KEEP_LOGITS} && cmd+=( --keep_logits )
${KEEP_EMBEDDINGS} && cmd+=( --keep_embeddings )
${DEBUG} && cmd+=( --debug ) && echo "DEBUG MODE ENABLED"
${GRAYSCALE} && cmd+=( --grayscale ) #pg_grayscale
${GAUSS_NOISE} && cmd+=( --gauss_noise )
${BLUR} && cmd+=( --blur )
${FORCE_OVERWRITE} && cmd+=( --force_overwrite ) && echo "FORCE OVERWRITE"

# Run command
"${cmd[@]}"