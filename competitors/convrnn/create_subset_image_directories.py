import pandas as pd
import os
import tarfile
from collections import defaultdict, Counter
import glob
import json
import shutil
import numpy as np
import argparse

_JSON_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required = True,
                        help="Root directory for ImageNet data (should be one level above train)")
    
    parser.add_argument("--experiment_root", type=str,required = True,
                        help="Directory where 16 class ImageNet subset image_names/ mappings has been saved")
    
    parser.add_argument("--new_dataset_root", type = str, required = True,
                        help = "Root directory where subsetted 16 class data will be stored")
    
    parser.add_argument("--split_idxs_root", type=str, required = True,
                      help="directory ")
    
    args = parser.parse_args()
    return args


def split_dataset(path_df, val_split=0.1, test_split=0, split_idxs_root=""):
  if not split_idxs_root:
    raise AssertionError("split_idxs_root not specified!")
    
  basename = f"{val_split}-{test_split}_val_test_split.json"
  split_path = os.path.join(split_idxs_root, basename)
  #if os.path.exists(split_path):
  #  print(f"Loading splic locs from {split_path}...")
  #  with open(split_path, "r") as infile:
  #    split_locs = json.load(infile)
  #else:
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



class Imagenet16Labels:
  def __init__(self, experiment_root):
    #self.savepath = savepath if save_path is not None else '/tmp/imagenet_labels.json'

    #self._download()
    self.experiment_root = experiment_root
    self._load_labels()


  def _load_labels(self):
    print(f"Loading data from {self.experiment_root}")

    self.cls_to_label = {}
    self.label_to_cls = {}
    self.train_classes = []

    #load different .txt files for 16 labels into separate 
    for file in os.listdir(os.path.join(self.experiment_root, 'image_names')):
        class_name = file[:-4]
        df = pd.read_csv(os.path.join(self.experiment_root,'image_names', file), names = ['full_file'])
        df['prefix'] = df['full_file'].apply(lambda x: x[:str.find(x,"_")]) 

        class_files = list(df['prefix'].unique())
        #self.label_to_cls[class_name] = class_files
        for n_file in class_files:
            self.cls_to_label[n_file] =  class_name
            self.label_to_cls[class_name] = n_file
            self.train_classes.append(n_file)


    self.num_classes = 16
    del df
    print("number of classes", self.num_classes)
    print("Fin.")
  

  def sample_classes(self, num_classes=10, target_labels=[]):
    if len(target_labels):
      sampled_labels = []
      for target_label in target_labels:
        sampled_labels += [lbl 
                           for lbl in self.labels 
                           if target_label.lower() in lbl.lower()]

    else:
      sampled_labels = list(
          np.random.choice(self.labels, num_classes, replace=False))
    
    # Get classes from labels
    sampled_classes = [self.label_to_cls[ele] for ele in sampled_labels]
    return sampled_labels, sampled_classes

  @property
  def labels(self):
    return list(self.label_to_cls.keys())

  @property
  def classes(self):
    return list(self.cls_to_label.keys())
  
  def lookup_cls(self, label):
    return self.label_to_cls[label]
  
  def lookup_lbl(self, cls):
    return self.cls_to_label[cls]


##Add necessary files vast directory for creating subset training and test dataset

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

def main(args):
  path_df = build_path_df(args.dataset_root, args.experiment_root)

  #Create or load train/test splits. We first load the full training population using path_df and then we apply the train/test split
  if os.path.exists(os.path.join(split_idxs_root, '0.1-0_val_test_split1.json')):
    print(f"loading split_locs from {os.path.join(split_idxs_root, '0.1-0_val_test_split.json')}", flush = True)
    split_locs = json.load(open(os.path.join(split_idxs_root, '0.1-0_val_test_split.json')))
  else:
    split_locs = split_dataset(path_df, val_split=0.1, test_split=0, split_idxs_root=split_idxs_root)

  #There are 1000 ImageNet subclasses that get aggregated into 16 super classes for purposes of this data subsetting. 
  # Therefore we need to map the subclasses to super classes

  #get list of superclass labels from 16class directory
  label_map = {}
  super_labels = [i[:-4] for i in os.listdir(os.path.join(args.experiment_root, 'image_names'))]
  print(f"Super labels: {super_labels}")
  for l in super_labels:
    #Create new subdirectories in new_dataset_root for train and val
    if os.path.exists(os.path.join(args.new_dataset_root, 'train', l)) == False:
          os.makedirs(os.path.join(args.new_dataset_root, 'train', l))
    if os.path.exists(os.path.join(args.new_dataset_root, 'val', l)) == False:
          os.makedirs(os.path.join(args.new_dataset_root, 'val', l))
    print(f"New train subdirectories: {os.listdir(os.path.join(args.new_dataset_root, 'train'))}")
    print(f"New val subdirectories: {os.listdir(os.path.join(args.new_dataset_root, 'val'))}")
    
    label_imgs = pd.read_csv(os.path.join(args.experiment_root, 'image_names',f'{l}.txt'), names = ['image_name'] )
    label_imgs['class'] = label_imgs['image_name'].apply(lambda x: x[:9])
    sub_classes = label_imgs['class'].unique().tolist()
    for sub in sub_classes:
      label_map[sub] = l

  train_df = path_df.loc[split_locs["train"]]
  val_df = path_df.loc[split_locs["val"]]
  
  print(f"# instances in training dataset: {len(train_df)}", flush = True)
  print(f"# instances in val dataset: {len(val_df)}",flush = True)

  train_df.sort_values(['class_id'], inplace = True)
  train_classes = train_df.class_id.unique()

  val_df.sort_values(['class_id'], inplace = True)
  val_classes = val_df.class_id.unique()

  
  for i,cls in enumerate(train_classes):
      print(f"CLASS {cls} # {i}: {cls}")
      cls_dir = os.path.join(args.new_dataset_root, 'train', label_map[cls])
      print("TRAINING",flush = True)
      print(f"Outputting to super directory : {cls_dir}",flush = True)
      sub_train_df = train_df[train_df['class_id']==cls]
      print(f"Adding {len(sub_train_df)} files to {cls_dir}", flush = True)

      for img_path in sub_train_df.path.tolist():
          img_file = img_path[str.rfind(img_path, '/')+1:]
          try:
            shutil.copyfile(img_path, os.path.join(cls_dir, img_file))
          except:
            print(f"Issue copying {img_path}")
      print(f"{len(os.listdir(cls_dir))} now in {cls_dir}",flush = True)

  for i, cls in enumerate(val_classes):
      print(f"CLASS {cls} # {i}: {cls}")
      cls_dir = os.path.join(args.new_dataset_root, 'val', label_map[cls])
      cls_dir = os.path.join(vast_dir, 'val', cls)
      
      print("VALIDATION",flush = True)
      print(f"Outputting to super directory : {cls_dir}",flush = True)
      sub_val_df = val_df[val_df['class_id']==cls]
      print(f"Adding {len(sub_val_df)} files to {cls_dir}",flush = True)

      for img_path in sub_val_df.path.tolist():
          img_file = img_path[str.rfind(img_path, '/')+1:]
          try:
            shutil.copyfile(img_path, os.path.join(cls_dir, img_file))
          except:
            print(f"Issue copying {img_path}")
      print(f"{len(os.listdir(cls_dir))} now in {cls_dir}",flush = True)


if __name__ == "__main__":
  args = setup_args()
  main(args)
         

