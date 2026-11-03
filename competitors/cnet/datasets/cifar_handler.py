"""CIFAR10/100 loader."""
import copy
import json
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from datasets import noise
import numpy as np

#Add gaussian noise class (from MSDNet)
class AddGaussianNoise(object):
    """
    Author: Omkar Kumbhar
    Description:
    Adding gaussian noise to images in the batch
    """
    def __init__(self, mean=0., std=1., contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast

    def __call__(self, tensor):
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = T.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean

class AllRandomNoise(object):
    def __init__(self, mean=0., std=0.04, contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast
        self.all_devs = np.arange(0.0,0.05,0.01)
    
    def __call__(self, tensor):
        self.std = np.random.choice(self.all_devs)
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = T.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean


class AddGaussianBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.std = std
    
    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = T.GaussianBlur(kernel_size = 7,sigma=self.std)(tensor)

        return tensor

class AllRandomBlur(object):
    def __init__(self, kernel=7):
        self.kernel = kernel
        self.all_devs = np.arange(0.0,1.0,0.1)
    
    def __call__(self, tensor):
        self.std = np.random.choice(self.all_devs)
        if self.std != 0.0:
            tensor = T.GaussianBlur(kernel_size = 7,sigma=self.std)(tensor)

        return tensor
        
class CIFAR10Handler(torchvision.datasets.CIFAR10):
  """CIFAR10 dataset handler."""

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class CIFAR100Handler(torchvision.datasets.CIFAR100):
  """CIFAR100 dataset handler."""

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


def get_transforms(
    dataset_key,
    mean,
    std,
    grayscale=False,
    gauss_noise=False,
    gauss_noise_std=0.0,
    blur=False,
    blur_std=0.0,
    noise_type=None,
    noise_transform_all=False
  ):
  """Create dataset transform list."""
  if dataset_key == "train":
    transforms_list = [
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip()
    ]
    if grayscale:
      transforms_list += [T.Grayscale(num_output_channels=3),
                        T.ToTensor()]
    if blur or gauss_noise:
      if not grayscale:
        transforms_list += [T.Grayscale(num_output_channels=3),
                            T.ToTensor()]
      
      if blur:
        transforms_list += [AllRandomBlur(7)]
      elif gauss_noise:
        transforms_list += [AllRandomNoise()]
    
    if (not grayscale) & (not blur) & (not gauss_noise):
      transforms_list += [
          T.ToTensor()
      ]
    print("TRANSFORMS_LIST", transforms_list)
    
  else:
    transforms_list = []
  
    #Add in grayscale, noise and blur handling for eval dataset
    if grayscale:
      transforms_list += [T.Grayscale(num_output_channels=3),
                        T.ToTensor()]
    print("BLUR", blur, blur_std)
    print("GAUSS_NOISE", gauss_noise, gauss_noise_std)
    if blur or gauss_noise:
      if not grayscale:
        transforms_list += [T.Grayscale(num_output_channels=3),
                            T.ToTensor()]
      
      if blur:
        transforms_list += [AddGaussianBlur(7,blur_std)]
      elif gauss_noise:
        transforms_list += [AddGaussianNoise(0.,gauss_noise_std)]
    
    if (not grayscale) & (not blur) & (not gauss_noise):
      transforms_list += [
          T.ToTensor()
        ]
      print("TRANSFORMS_LIST", transforms_list)

  if (noise_type is not None
      and (dataset_key == "train" or noise_transform_all)):
    transforms_list.append(noise.NoiseHandler(noise_type))

  transforms = T.Compose(transforms_list)

  return transforms


def split_dev_set(
    dataset_src, 
    mean, 
    std, 
    dataset_len, 
    val_split, 
    split_idxs_root, 
    load_previous_splits, 
    verbose=False
  ):
  """Load or create (and save) train/val split from dev set."""
  # Compute number of train / val splits
  n_val_samples = int(dataset_len * val_split)
  n_sample_splits = [dataset_len - n_val_samples, n_val_samples]

  # Split data
  train_set, val_set = torch.utils.data.random_split(dataset_src,
                                                     n_sample_splits)

  train_set.dataset = copy.copy(dataset_src)
  val_set.dataset.transform = get_transforms("test", mean, std)

  # Set indices save/load path
  val_percent = int(val_split * 100)
  if ".json" not in split_idxs_root:
    idx_filepath = os.path.join(
        split_idxs_root, 
        f"{val_percent}-{100-val_percent}_val_split.json"
    )
  else:
    idx_filepath = split_idxs_root

  # Check load indices
  if load_previous_splits and os.path.exists(idx_filepath):
    if verbose:
      print(f"Loading previous splits from {idx_filepath}")
    with open(idx_filepath, "r") as infile:
      loaded_idxs = json.load(infile)

    # Set indices
    train_set.indices = loaded_idxs["train"]
    val_set.indices = loaded_idxs["val"]

  # Save idxs
  else:
    if verbose:
      print(f"Saving split idxs to {idx_filepath}...")
    save_idxs = {
        "train": list(train_set.indices),
        "val": list(val_set.indices),
    }

    # Dump to json
    with open(idx_filepath, "w") as outfile:
      json.dump(save_idxs, outfile)

  if verbose:
    # Print
    print(f"{len(train_set):,} train examples loaded.")
    print(f"{len(val_set):,} val examples loaded.")

  return train_set, val_set


def set_dataset_stats(dataset_name):
  """Set dataset stats for normalization given dataset."""
  if "cifar10" in dataset_name.lower():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

  elif dataset_name.lower() == "cifar100":
    mean = (0.5071, 0.4866, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
  return mean, std


def build_dataset(
    root,
    dataset_name,
    dataset_key,
    mean,
    std,
    grayscale=False,
    gauss_noise=False,
    gauss_noise_std=0.0,
    blur=False,
    blur_std=0.0,
    val_split=None,
    split_idxs_root=None,
    load_previous_splits=True,
    noise_type=None,
    noise_transform_all=False,
    verbose=True,
  ):
  """Build dataset."""
  if verbose:
    print(f"Loading {dataset_name} {dataset_key} data...")

  # Datsaet
  if (dataset_name.lower() == "cifar10") or (dataset_name.lower() == "cifar10_normalized"):
    dataset_op = CIFAR10Handler
  elif dataset_name.lower() == "cifar100":
    dataset_op = CIFAR100Handler
  else:
    assert False, f"{dataset_name} wrapper not implemented!"

  # Transforms
  transforms = get_transforms(
    dataset_key, 
    mean, 
    std, 
    grayscale,
    gauss_noise,
    gauss_noise_std,
    blur,
    blur_std,
    noise_type, 
    noise_transform_all,
  )

  # Build dataset source
  dataset_src = dataset_op(
    root=root,
    train=dataset_key == "train",
    transform=transforms,
    target_transform=None,
    download=True,
  )
  print("dataset_src attributes")
  print(dir(dataset_src))
  print(type(dataset_src.data[0]))
  print(dataset_src.data.shape)

  # Get number samples in dataset
  dataset_len = dataset_src.data.shape[0]

  # Split
  if dataset_key == "train":
    if val_split:
      dataset_src = split_dev_set(
        dataset_src,
        mean,
        std,
        dataset_len,
        val_split,
        split_idxs_root,
        load_previous_splits,
        verbose=verbose
      )
    else:
      dataset_src = dataset_src, None

  # Stdout out
  if verbose:
    dataset_key_str = "dev" if dataset_key=="train" else dataset_key
    print((f"{dataset_len:,} "
           f"{dataset_key_str} "
           f"examples loaded."))

  return dataset_src


def create_datasets(
    root,
    dataset_name,
    val_split,
    grayscale=False,
    gauss_noise=False,
    gauss_noise_std=0.0,
    blur=False,
    blur_std=0.0,
    load_previous_splits=False,
    split_idxs_root=None,
    noise_type=None,
    verbose=False
  ):
  """Create train, val, test datasets."""

  # Set stats
  mean, std = set_dataset_stats(dataset_name)

  print("CREATE_DATASETS")
  print("GRAYSCALE", grayscale)
  print("GAUSS_NOISE", gauss_noise)
  print("GAUSS_NOISE_STD", gauss_noise_std)
  print("BLUR", blur)
  print("BLUR_STD", blur_std)
  # Build datasets
  train_dataset, val_dataset = build_dataset(
    root,
    dataset_name,
    dataset_key="train",
    mean=mean,
    std=std,
    grayscale=grayscale,
    gauss_noise=gauss_noise,
    gauss_noise_std=gauss_noise_std,
    blur=blur,
    blur_std=blur_std,
    val_split=val_split,
    split_idxs_root=split_idxs_root,
    load_previous_splits=load_previous_splits,
    noise_type=noise_type,
    verbose=verbose
  )

  test_dataset = build_dataset(
    root,
    dataset_name,
    dataset_key="test",
    mean=mean,
    std=std,
    grayscale=grayscale,
    gauss_noise=gauss_noise,
    gauss_noise_std=gauss_noise_std,
    blur=blur,
    blur_std=blur_std,
    noise_type=noise_type,
    load_previous_splits=load_previous_splits
  )

  # Package
  dataset_dict = {
      "train": train_dataset,
      "val": val_dataset,
      "test": test_dataset,
  }

  return dataset_dict
