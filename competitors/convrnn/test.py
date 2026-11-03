import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from PIL import Image, ImageEnhance
import cv2
import urllib
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import glob
from random import shuffle
import h5py
import torch
from torchvision import transforms
import math
import time
import json
import os
import argparse
import pandas as pd

# tf.enable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from rcnn_sat import preprocess_image, bl_net
from load_data import load_dataset, load_dataset_h5, prep_pixels, prep_pixels_h5, load_dataset_ImageNet_test
from custom_transforms import *

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', required = True, type = str,
                        help = 'Root directory where human test experimental data has been downloaded')
    parser.add_argument('--model_dir', required = True, 
                        help = 'Directory where models outputs will be saved')
    parser.add_argument('--tag', required = True, type=str,
                        help = 'label for type of image perturbation/ experiment that is being run: color, grayscale, blur-color, noise-contrast-gray')
    
    parser.add_argument('--batch_size', default = 128, type = int)

    parser.add_argument('--pretrained', default=True, type=bool,
                        help = 'whether to use pretrained bl model')

    args = parser.parse_args()
    return args

def main(args):
    if args.tag == 'color':
        test_dir = os.path.join(args.test_data_dir, 'color')
    elif args.tag == 'grayscale':
        test_dir = os.path.join(args.test_data_dir, 'grayscale')
    elif args.tag == 'blur-color':
        test_dir = os.path.join(args.test_data_dir, 'blur_color')
    elif args.tag == 'noise-contrast-gray':
        test_dir = os.path.join(args.test_data_dir, 'noise-contrast-gray')
    
    #Load test data
    test_ds = load_dataset_ImageNet_test(test_dir = test_dir, batch_size = args.batch_size)
    
    #Base Data Augmentation
    test_data_augmentation = tf.keras.Sequential([layers.Rescaling(1./255),
                                                 layers.Resizing(224, 224, crop_to_aspect_ratio = True)])
    
    #Grayscale add data augmentation
    if args.tag == 'grayscale':
        test_data_augmentation.add(GrayscaleChannels())
    
        
    test_ds = test_ds.map(lambda x, y: (test_data_augmentation(x),y))
    
    #Get data in format model expects
    test_ds = test_ds.unbatch()
    testX = np.array(list(test_ds.map(lambda x, y: x)))
    testy = np.array(list(test_ds.map(lambda x, y: y)))
    
    
    #Set up model
    input_layer = tf.keras.layers.Input((224, 224, 3))
    model = bl_net(input_layer, classes=16, cumulative_readout=False)
        
    os.chdir(args.model_dir)
    model.load_weights(os.path.join('model/ckpt','pretrained_mp_color-gray.hdf5'),
                        skip_mismatch=True,by_name=True)

        ## Lets try fine tuning it
        # tf.keras.utils.plot_model(model,to_file='check.png')

    skip_layers = ['ReadoutDense','Sotfmax_Time_0','Sotfmax_Time_1',
                'Sotfmax_Time_2','Sotfmax_Time_3','Sotfmax_Time_4',
                'Sotfmax_Time_5','Sotfmax_Time_6','Sotfmax_Time_7']

    for layer in model.layers:
        if layer.name in skip_layers:
            layer.trainable = True
        else:
            layer.trainable = False

    # compile model with optimizer and loss
    """
    B, BL and parameter-matched controls (B-K, B-F and B-D) were trained for a total of 90 epochs 
    with a batch size of 100. B-U was trained using the same procedure but with a batch size of 64 
    due to its substantially larger number of parameters.

    The cross-entropy between the softmax of the network category readout and the labels 
    was used as the training loss. For networks with multiple readouts (BL and B-U), 
    we calculate the cross-entropy at each readout and average this across readouts. 
    Adam [64] was used for optimisation with a learning rate of 0.005 and epsilon parameter 0.1. 
    L2-regularisation was applied throughout training with a coefficient of 10−6.

    """
    cce = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    trialX = np.zeros((len(testX),224,224,3))
    for i,image in enumerate(testX):
        trialX[i] = image

    #Create predictions dataframe
    file_paths = []
    file_names = []

    for i, label in enumerate(sorted(os.listdir(test_dir))):
        file_list = sorted(os.listdir(os.path.join(test_dir, label)))
        for f in file_list:
            file_paths.append(os.path.join(test_dir, label, f))
            file_names.append(f)
    
    true_labels = np.argmax(testy, axis =1)

    d = {'file_paths': file_paths, 
        'file_names': file_names,
        'labels': true_labels}

    preds_df = pd.DataFrame(d)

    scores = model.predict(trialX)
    for i, score in enumerate(scores):
        predictions = np.argmax(score, axis =1)
        preds_df[f'predictions_exit{i}'] = predictions

    preds_df.to_csv(os.path.join(args.model_dir, 'model/results', f'predictions_{args.tag}.csv'))

if __name__ == "__main__":
  args = setup_args()
  main(args)
