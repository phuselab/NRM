import numpy as np
from torchvision import transforms
import torch
import tensorflow_io as tfio
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow.image import random_flip_left_right, random_crop, central_crop, adjust_contrast

def standard_transforms(image, train):
    if train:
        image = random_flip_left_right(image)
        image = random_crop(image, (224, 224))
    else:
        image = central_crop(image, image.shape[1]/224)
    return image

def all_random_blur(image, kernel=49, std=3.0):
    print(tf.test.gpu_device_name())
    device = tf.test.gpu_device_name()
    with tf.device(f'{device}'):
        #image = random_flip_left_right(image)
        #image = random_crop(image, (128, 224, 224, image.shape[3]))
        all_devs = np.arange(0.0, std+0.1, 0.1)
        std = np.random.choice(all_devs)

        if std != 0.0:
            image = tf.transpose(image, perm = [0,3,1,2])
            image = tfio.experimental.filter.gaussian(image, ksize = (kernel, kernel), sigma=std)
            image = tf.transpose(image,perm= [0,2,3,1])
        
        return image

class AllRandomBlur(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return all_random_blur(x)

def add_gaussian_blur(image, kernel=49, std=3.0):
    print(tf.test.gpu_device_name())
    device = tf.test.gpu_device_name()
    with tf.device(f'{device}'):
        if std != 0.0:
            image = tf.transpose(image, perm = [0,3,1,2])
            image = tfio.experimental.filter.gaussian(image, ksize = kernel, sigma=std)
            image = tf.transpose(image,perm= [0,2,3,1])
        
        return image

class AddGaussianBlur(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return add_gaussian_blur(x)

def all_random_noise(image, std=0.04, mean=0.0, contrast=0.2):
    print(tf.test.gpu_device_name())
    device = tf.test.gpu_device_name()
    with tf.device(f'{device}'):
        if image.shape[3]!=3:
            image = tf.image.grayscale_to_rgb(image)
        print("image", image.shape)
        std = np.random.choice(np.arange(0.0, std+0.01, 0.01))
        n = 128* image.shape[1] * image.shape[2]
        sd2 = std * 2
        noise = np.array([])
        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = np.random.randn(m) * std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = np.concatenate((noise, new))
        
        # pick first n samples and reshape to 2D
        noise = np.reshape(noise[:n], (128, image.shape[1], image.shape[2]))
        # stack noise and translate by mean to produce std + 
        newnoise = tf.cast(tf.convert_to_tensor(tf.stack([noise, noise, noise], axis=3)+ mean), tf.float32)
        #image = tf.stack([image, image, image], axis=3)

        image = tf.add(image, tf.reshape(0.5 - tf.reduce_mean(image, axis = (1,2,3)),(128,1,1,1)))

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        image = tf.transpose(image, perm = [0,3,1,2])
        image = adjust_contrast(image, contrast)
        image = tf.transpose(image,perm= [0,2,3,1])
        return image + newnoise + mean

class AllRandomNoise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return all_random_noise(x)

def add_gaussian_noise(image, std=0.04, mean=0, contrast=0.2):
    print(tf.test.gpu_device_name())
    device = tf.test.gpu_device_name()
    with tf.device(f'{device}'):
        if image.shape[3]!=3:
            image = tf.image.grayscale_to_rgb(image)
        noise = np.array([])
        n = 128* image.shape[1] * image.shape[2] * image.shape[3]
        sd2 = std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = np.random.randn(m) * std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = np.concatenate((noise, new))
        
    # pick first n samples and reshape to 2D
        noise = np.reshape(noise[:n], (128, image.shape[1], image.shape[2]))
        # stack noise and translate by mean to produce std + 
        newnoise = tf.cast(tf.convert_to_tensor(tf.stack([noise, noise, noise], axis=3)+ mean), tf.float32)
        #image = tf.stack([image, image, image], axis=3)
        print("newnoise", newnoise.shape)

        # shift image hist to mean = 0.5
        image = tf.add(image, tf.reshape(0.5 - tf.reduce_mean(image, axis = (1,2,3)),(128,1,1,1)))

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        image = tf.transpose(image, perm = [0,3,1,2])
        image = adjust_contrast(image, contrast)
        image = tf.transpose(image,perm= [0,2,3,1])
        
        return image + newnoise + mean

class AddGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return all_random_noise(x)

def grayscale_channels(image):
    device = tf.test.gpu_device_name()
    with tf.device(f'{device}'):
        if image.shape[3]!=3:
            image = tf.image.grayscale_to_rgb(image)
        print(image.shape)
        return image

class GrayscaleChannels(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, x):
        return grayscale_channels(x)
