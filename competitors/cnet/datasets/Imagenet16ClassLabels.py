import os
import urllib
import json
import numpy as np
import pandas as pd

_JSON_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


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

    #data = {int(key): val for key, val in data.items()}
    #self.idx_to_class = {key: val[0] for key, val in data.items()}
    #self.idx_to_label = {key: val[1] for key, val in data.items()}

    #self.cls_to_idx = {val[0]: key for key, val in data.items()}
    #self.label_to_idx = {val[1]: key for key, val in data.items()}

    #self.cls_to_label = {val[0]: val[1] for key, val in data.items()}
    #self.label_to_cls = {val: key for key, val in self.cls_to_label.items()}

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