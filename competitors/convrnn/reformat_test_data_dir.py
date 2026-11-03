import shutil, os, sys
import pandas as pd
import numpy as np
import argparse

old_dir = '/scratch/sbp354/SAT_human_data'
new_dir = '/scratch/sbp354/SAT_human_data_convrnn'
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_dir", type=str, required = True,
                        help="Root directory for data collected in human experiments")
    
    parser.add_argument("--new_dir", type=str, required = True,
                        help="New directory for data collected in human experiments formatted for convRNN TensorFlow dataloader")
    args = parser.parse_args()
    return args

def main(args):

    classes = ['airplane',  'bear',  'bicycle',  'bird',  'boat',  'bottle',  'car',  'cat',  'chair',
            'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    new_exp_types = ['color', 'grayscale', 'blur_color', 'noise_grayscale']

    new_old_exp_map = {'color': 'ColorGraySplit', 
                    'grayscale':'ColorGraySplit',
                    'blur_color':'BlurSplit_color',
                    'noise_grayscale': 'NoiseSplit_gray_contrast0.2'}
    old_exp_types = ['ColorGraySplit', 'BlurSplit_color', 'NoiseSplit_gray_contrast0.2']

    if os.path.exists(new_dir)==False:
        os.mkdir(new_dir)

    for exp in new_exp_types:
        print(f"COPYING OVER FILES FOR {exp}")
        ##Copy over color files
        old_exp_dir = os.path.join(args.old_dir, new_old_exp_map[exp])
        new_exp_dir = os.path.join(args.new_dir, exp)
        if os.path.exists(new_exp_dir)==False:
            os.mkdir(new_exp_dir)
            print(f"Making directory: {new_exp_dir}")

        for cls in classes:
            if os.path.exists(os.path.join(new_exp_dir, cls))==False:
                os.mkdir(os.path.join(new_exp_dir, cls))
                print(f"Making directory {os.path.join(new_exp_dir, cls)}")

        for i in range(5):
            exit_dir = os.path.join(old_exp_dir, str(i))
            if exp == 'color':
                img_file_list = [f for f in os.listdir(exit_dir) if str.find(f, "_c_")>=0]
                print(img_file_list)
            elif exp == 'grayscale':
                img_file_list = [f for f in os.listdir(exit_dir) if str.find(f, "_g_")>=0]
                print(img_file_list)
            else:
                img_file_list = os.listdir(exit_dir)

            for f in img_file_list:
                old_file_path = os.path.join(exit_dir, f)
                for cls in classes:
                    if str.find(old_file_path, cls) > -1:
                        new_file_name = str.replace(f, '.JPEG', f'_exit{i}.JPEG')
                        new_file_path = os.path.join(new_exp_dir, cls, new_file_name)

                        shutil.copy(old_file_path, new_file_path)

if __name__ == "__main__":
  args = setup_args()
  main(args)




