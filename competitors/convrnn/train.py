import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from PIL import Image, ImageEnhance
import cv2
import urllib
import numpy as np
from tensorflow.keras.utils import to_categorical
import glob
from random import shuffle
import h5py
import torch
from torchvision import transforms
import math
import time
import os
import argparse

from rcnn_sat import preprocess_image, bl_net
from load_data import load_dataset, load_dataset_h5, prep_pixels, prep_pixels_h5, load_dataset_ImageNet
from custom_transforms import *

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
#default='noise-contrast-gray'
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required = True, type = str,
                        help = 'Root directory where 16class ImageNet data is stored')
    parser.add_argument('--model_dir', required = True, 
                        help = 'Directory where models outputs will be saved')
    parser.add_argument('--tag', required = True, type=str,
                        help = 'label for type of image perturbation/ experiment that is being run: color, grayscale, blur-color, noise-contrast-gray')
    
    parser.add_argument('--batch_size', default = 128, type = int)
    parser.add_argument('--epochs', default = 100, type = int)

    parser.add_argument('--pretrained', default=True, type=bool,
                        help = 'whether to use pretrained bl model')

    args = parser.parse_args()
    return args

def main(args):

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        #Load ImageNet data
        if args.tag == 'color':
            grayscale = blur = gauss_noise = False
        elif args.tag == 'grayscale':
            grayscale = True
            blur = gauss_noise = color = False
        elif args.tag == 'blur-color':
            grayscale = gauss_noise = False
            blur = True
        elif args.tag == 'noise-contrast-gray':
            grayscale = gauss_noise = True
            blur = False
        train_ds, test_ds, class_weights = load_dataset_ImageNet(train_dir = os.path.join(args.data_dir, 'train_16classes'),
                                                                 val_dir = os.path.join(args.data_dir, 'val_16classes'),
                                                                    batch_size = args.batch_size,
                                                                    grayscale = grayscale,
                                                                    blur = blur,
                                                                    gauss_noise = gauss_noise)
        train_ds = train_ds.take(int(len(train_ds)-1))
        test_ds = train_ds.take(int(len(test_ds)-1))
        
        #Model Set-up
        input_layer = tf.keras.layers.Input((224, 224, 3))
        model = bl_net(input_layer, classes=16, cumulative_readout=False)

        os.chdir(args.model_dir)
        if args.pretrained:
            model.load_weights('bl_imagenet.h5',skip_mismatch=True,by_name=True)

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

        ckpt_path = os.path.join(args.model_dir, 'model/ckpt')
        if os.path.exists(ckpt_path)==False:
            os.mkdir(ckpt_path)
        
        #Set up model checkpoint
        checkpoint = ModelCheckpoint(os.path.join(ckpt_path, "pretrained_mp_noise_{}_new.hdf5".format(args.tag)),
                                        monitor='loss', verbose=1,
                                        save_best_only=True, mode='auto', period=1)

        ##Data Augmentation - different perturbations require different data transforms
        
        #Base train and test data augmentations
        train_data_augmentation = tf.keras.Sequential([layers.Rescaling(1./255),
                                            layers.RandomFlip(mode = 'horizontal'),
                                            layers.RandomCrop(224, 224)])
        test_data_augmentation = tf.keras.Sequential([layers.Rescaling(1./255)])
        
        #Grayscale-specific augmentation
        if grayscale:
            train_data_augmentation.add(GrayscaleChannels()) 
            test_data_augmentation.add(GrayscaleChannels())

        #Gaussian Noise specific augmentation
        elif gauss_noise:
            train_data_augmentation.add(AllRandomNoise())
            test_data_augmentation.add(AddGaussianNoise())

        #Blur specific augmentation
        elif blur:
            train_data_augmentation.add(AllRandomBlur())
            test_data_augmentation.add(AddGaussianBlur())

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.map(lambda x, y: (train_data_augmentation(x, training = True), y))
        train_ds =train_ds.prefetch(buffer_size=AUTOTUNE)

        test_ds = test_ds.map(lambda x, y: (test_data_augmentation(x), y))

        #Fit model
        history = model.fit(x=train_ds,
                            validation_data=test_ds,
                            steps_per_epoch=len(train_ds),
                            epochs=args.epochs,
                            callbacks=[checkpoint],
                            class_weight = class_weights)

        #Save model 
        model.save(os.path.join('model','{}_{}'.format(
            args.tag,
            time.strftime('%Y.%m.%d_%H.%M.%S')))
        )
if __name__ == "__main__":
  args = setup_args()
  main(args)
