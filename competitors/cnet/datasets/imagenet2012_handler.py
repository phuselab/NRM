import os
import glob
import json
from tkinter import E
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from PIL import Image
from datasets.Imagenet16ClassLabels import Imagenet16Labels
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import time


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def split_dataset(path_df, val_split=0.1, test_split=0.1, split_idxs_root=""):
  if not split_idxs_root:
    raise AssertionError("split_idxs_root not specified!")
    
  basename = f"{val_split}-{test_split}_val_test_split.json"
  split_path = os.path.join(split_idxs_root, basename)
  split_locs = defaultdict(list)
  for class_lbl, class_df in path_df.groupby("class_lbl"):
    n_samples = len(class_df)
    n_test = int(n_samples * test_split)
    n_dev = n_samples - n_test
    n_val = int(n_dev * val_split)

    test_locs = list(class_df.iloc[:n_test].index)
    val_locs = list(class_df.iloc[n_test:n_test+n_val].index)
    train_locs = list(class_df.iloc[n_test+n_val:].index)

    split_locs["test"] += test_locs
    split_locs["val"] += val_locs
    split_locs["train"] += train_locs
    
  # Save
  print(f"Saving split locs to {split_path}...")
  with open(split_path, "w") as outfile:
    json.dump(split_locs, outfile)
  return split_locs


def subset_path_df(path_df, params):
  target_classes = params.get("target_classes", [])
  if target_classes:
    subset_dfs = []
    for target_class in target_classes:
      df_i = path_df[path_df.class_lbl.str.contains(target_class)]
      subset_dfs.append(df_i)
    path_df = pd.concat(subset_dfs)
  
  n_subset_classes = path_df.class_lbl.unique().shape[0]
  max_classes = params.get("max_classes", 0)
  print(f"n_subset_classes: {n_subset_classes}")
  print(f"max_classes: {max_classes}")

  if max_classes and max_classes < n_subset_classes:
    cls_counts = Counter(path_df.class_lbl)
    count_df = pd.DataFrame({
      "class_lbl": list(cls_counts.keys()), 
      "count": list(cls_counts.values())
    })
    count_df = count_df.sort_values("count", ascending=False)
    selected_classes = list(count_df.iloc[:max_classes].class_lbl)
    path_df = path_df[path_df.class_lbl.isin(selected_classes)]
  
  # Set class label
  class_labels = np.sort(path_df.class_lbl.unique())
  lookup = {lbl: i for i, lbl in enumerate(class_labels)}
  path_df["y"] = [lookup[class_lbl] for class_lbl in path_df.class_lbl]
  return path_df

def build_path_df(root, experiment_root, subdir='train'):
  # Label handler
  label_handler = Imagenet16Labels(experiment_root)
  
  # Set path
  dataset_path = os.path.join(root, subdir)
  
  # Class dirs
  class_dirs = glob.glob(f'{dataset_path}/*')
  df_dict = defaultdict(list)
  for class_dir in class_dirs:
    class_id = class_dir[str.find(class_dir, "/n")+1:]
    if class_id in(label_handler.train_classes):
      #class_id = os.path.basename(class_dir)
      class_lbl = label_handler.lookup_lbl(class_id)

      img_paths = glob.glob(f"{class_dir}/*")
      for img_path in img_paths:
        df_dict["class_lbl"].append(class_lbl)
        df_dict["class_id"].append(class_id)
        df_dict["path"].append(img_path)
  path_df = pd.DataFrame(df_dict)
  return path_df


def build_test_path_df(test_data_root, grayscale, gauss_noise, blur, gauss_noise_std, blur_std):
  if gauss_noise:
    if grayscale:
      dataset_path = os.path.join(test_data_root, 'NoiseSplit_gray_contrast0.2')
    else:
      dataset_path = os.path.join(test_data_root, 'NoiseSplit_color')
  elif blur:
    if grayscale:
      dataset_path = os.path.join(test_data_root, 'BlurSplit_gray')
    else:
      dataset_path = os.path.join(test_data_root, 'BlurSplit_color')
  else:
    dataset_path = os.path.join(test_data_root, 'ColorGraySplit')

  df_dict = defaultdict(list)
  for i in range(5):
    if blur:
      if str(blur_std) not in([str(i) for i in [0, 3, 9]]):
        test_paths = glob.glob(f'{dataset_path}/{i}/*blur_*')
      else:
        test_paths = glob.glob(f'{dataset_path}/{i}/*blur_{blur_std}*')
    elif gauss_noise:
      if str(gauss_noise_std) not in([str(i) for i in [0, 0.04, 0.16]]):
        test_paths = glob.glob(f'{dataset_path}/{i}/*noise_*')
      else:
        test_paths = glob.glob(f'{dataset_path}/{i}/*noise_{gauss_noise_std}_*')
    else:
      test_paths = glob.glob(f'{dataset_path}/{i}/*')

    for path in test_paths:
      if not gauss_noise and not blur:
        if grayscale:
          if str.find(path, "gray"):
            file_name = str.split(path, "/")[-1]
            class_lbl = str.split(file_name, "_")[-1][:-5]

            df_dict['class_lbl'].append(class_lbl)
            df_dict['timestep'].append(i)
            df_dict['path'].append(path)
        else:
            file_name = str.split(path, "/")[-1]
            class_lbl = str.split(file_name, "_")[-1][:-5]

            df_dict['class_lbl'].append(class_lbl)
            df_dict['timestep'].append(i)
            df_dict['path'].append(path)
      else:
        file_name = str.split(path, "/")[-1]
        class_lbl = str.split(file_name, "_")[-1][:-5]

        df_dict['class_lbl'].append(class_lbl)
        df_dict['timestep'].append(i)
        df_dict['path'].append(path)

  path_df = pd.DataFrame(df_dict)
  return path_df

    
def split_dev_set(img_paths, val_split):
  class_paths = defaultdict(list)
  for idx, im_path in enumerate(img_paths):
    class_id = os.path.basename(os.path.dirname(im_path))
    class_paths[class_id].append((idx, im_path))

  dataset_idx_lookup = defaultdict(list)
  for class_key, vals in class_paths.items():
    vals = np.array(vals)
    idxs = vals[:, 0].astype(np.int)
    paths = vals[:, 1]

    n_val_samples = int(len(idxs) * val_split)
    val_idxs = np.random.choice(idxs, n_val_samples, replace=False)
    train_idxs = list(set(idxs).difference(set(val_idxs)))
    dataset_idx_lookup['val'] += list(val_idxs)
    dataset_idx_lookup['train'] += list(train_idxs)

  dataset_idx_lookup = {
      key: np.array(val)
      for key, val in dataset_idx_lookup.items()
  }
  return dataset_idx_lookup


class EnforceShape:
  """ Catches and converts grayscale and RGBA --> RGB """
  def __call__(self, x):
    if x.shape[0] == 1:
      x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
      x = x[:3]
    return x

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
        tensor = torch.unsqueeze(tensor, dim = 0)
        tensor = T.functional.adjust_contrast(tensor, self.contrast)
        
        return torch.squeeze(tensor,0) + newnoise + self.mean

class AllRandomNoise(object):
    def __init__(self, mean=0., std=0.04, contrast=0.2):
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
        tensor = torch.unsqueeze(tensor, dim = 0)
        tensor = T.functional.adjust_contrast(tensor, self.contrast)
        return torch.squeeze(tensor,0) + newnoise + self.mean


class AddGaussianBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.std = std
    
    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = T.GaussianBlur(kernel_size = 7,sigma=self.std)(tensor)

        return tensor

class AllRandomBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.all_devs = np.arange(0.0,1.0,0.1)
    
    def __call__(self, tensor):
        self.std = np.random.choice(self.all_devs)
        if self.std != 0.0:
            tensor = T.GaussianBlur(kernel_size = self.kernel,sigma=self.std)(tensor)

        return tensor


class ImagenetDataset(Dataset):
  def __init__(self, path_df, dataset_key, grayscale, gauss_noise, gauss_noise_std, blur, blur_std): #pg_grayscale
    self.path_df = path_df
    self.dataset_key = dataset_key
    self.grayscale = grayscale
    self.gauss_noise = gauss_noise
    self.gauss_noise_std = gauss_noise_std
    self.blur = blur
    self.blur_std = blur_std

    # Setup transforms and paths
    self._setup_transforms()

  def _setup_transforms(self):
    if self.dataset_key == 'test_human':
      xform_list = [T.ToTensor(),
                    EnforceShape()]
    elif self.dataset_key == 'train':
      xform_list = [
        T.RandomResizedCrop(224), 
        T.RandomHorizontalFlip(),
      ]

      #Add in grayscale, noise and blur handling if necessary
      if self.grayscale: #pg_grayscale
        xform_list += [
                      T.Grayscale(num_output_channels=3),
                      T.ToTensor(),
                        EnforceShape()] #pg_grayscale
        if self.blur:
          xform_list += [AllRandomBlur(kernel=49, std=3.0),
                      EnforceShape()]
        elif self.gauss_noise:
          xform_list += [AllRandomNoise(),
                        EnforceShape()]
      else:
        if self.blur:
          xform_list += [T.ToTensor(),
                        AllRandomBlur(kernel=49, std=3.0),
                        EnforceShape()]
        elif self.gauss_noise:
          xform_list += [T.ToTensor(),
                        EnforceShape(),
                        AllRandomNoise()]
        else:
          xform_list += [
                          T.ToTensor(),
                          EnforceShape()
                          ]
    
    else:
      xform_list = [T.Resize(224), T.CenterCrop(224)]
      #Add in grayscale, noise and blur handling if necessary
      if self.grayscale: #pg_grayscale
        xform_list+= [T.Grayscale(num_output_channels=3),
                      T.ToTensor(),
                      EnforceShape()]#pg_grayscale
        if self.blur:
          xform_list += [AddGaussianBlur(49,self.blur_std),
                       EnforceShape()]
        elif self.gauss_noise:
          xform_list += [AddGaussianNoise(0.,self.gauss_noise_std),
                        EnforceShape()]
      else:
        xform_list += [
                        T.ToTensor(),
                        EnforceShape()
                        ]
        if self.blur:
          xform_list += [AddGaussianBlur(49,self.blur_std),
                       EnforceShape()]
        elif self.gauss_noise:
          xform_list += [EnforceShape(),
                         AddGaussianNoise(0.,self.gauss_noise_std)]

    self.transforms = T.Compose(xform_list)

  def __len__(self):
    return len(self.path_df)

  def __getitem__(self, idx):
    # Load df
    df_i = self.path_df.iloc[idx]
               
    # Load img path
    path = df_i.get("path")

    # Load image
    img = Image.open(path)
  
    # Apply image transforms
    img = self.transforms(img)

    # Get label and y
    label = df_i.get("class_lbl")
    y = df_i.get("y")

    return img, y  #, label

 
def create_datasets(path_df, val_split, test_split, split_idxs_root, experiment_root, grayscale, gauss_noise, gauss_noise_std, blur, blur_std, test_path_df): #pg_grayscale
  # Label handler
  label_handler = Imagenet16Labels(experiment_root)
  # Make sure split idx root exists
  if not os.path.exists(split_idxs_root):
    print(f"Creating {split_idxs_root}...")
    os.makedirs(split_idxs_root)
  
  # Split data
  split_locs = split_dataset(path_df, val_split, test_split, split_idxs_root)

  # Train
  print("Loading train data...")
  train_df = path_df.loc[split_locs["train"]]
  train_dataset = ImagenetDataset(train_df, 'train', grayscale, gauss_noise, gauss_noise_std, blur, blur_std) #pg_grayscale
  print(f"{len(train_dataset):,} train examples loaded.")

  # Validation
  print("Loading validation data...")
  val_df = path_df.loc[split_locs["val"]]
  val_dataset = ImagenetDataset(val_df, 'val',grayscale,gauss_noise, gauss_noise_std,blur, blur_std) #pg_grayscale
  print(f"{len(val_dataset):,} train examples loaded.")

  # Test
  print("Loading test data...")
  test_df = path_df.loc[split_locs["test"]]
  test_dataset = ImagenetDataset(test_df, 'test',grayscale, gauss_noise, gauss_noise_std,blur, blur_std) #pg_grayscale
  print(f"{len(test_dataset):,} test examples loaded.")

  # Test human data
  print("Loading human test data...")
  test_human_dataset = ImagenetDataset(test_path_df, 'test_human',grayscale, gauss_noise, gauss_noise_std,blur, blur_std) 
  print(f"{len(test_human_dataset):,} human test examples loaded.")
  

  # Package
  dataset_dict = {
      "train": train_dataset,
      "val": val_dataset,
      "test": test_dataset,
      "test_human": test_human_dataset
  }
  print(f"Complete.")
  return dataset_dict