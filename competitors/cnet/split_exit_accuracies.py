import pandas as pd
import os
import torch
import json
import numpy as np
import argparse

#experiments_dir = '/scratch/sbp354/cascade_output/experiments/resnet18_ImageNet2012_16classes_rebalanced/experiments'
#human_data_dir = '/scratch/sbp354/SAT_human_data'
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_root", type=str,
                        help="root directory where model outputs have been saved")
    
    parser.add_argument("--human_data_dir", type = str, 
                        help = "directory where human experimental image data has been stored")

    parser.add_argument("--output_dir", type = str, 
                        help = "directory where accuracy outputs and results are written out")
    
    parser.add_argument("--dataset_name", type=str, default="ImageNet2012_16classes_rebalanced",
                      help="Dataset name: CIFAR10, CIFAR100, TinyImageNet")

    parser.add_argument("--models", type = str, default = ['(1.0),parallel', '(0.0),serial'],
                        help = "base model names that are being evaluated. take the format (lambda),cascaded_scheme")

    parser.add_argument("--get_acc_by_cat", type = bool, action = argparse.BooleanOptionalAction, default = False,
                        help = "Whether or not to parse accuracies by different image categories. This is currently only set up to run for color images")
    
    args = parser.parse_args()
    return args

def get_evens(x):
    return x % 2 == 0
def get_odds(x):
    return x %2 != 0

def create_df(targets, preds, model, experiment, exit, std= None):
    '''
    Function to create output dataframe of accuracies. 

    Inputs:
    ------
        targets : np.array of true labels
        preds : np.array of predictions from a given model
        experiment : str, model key for results being processed
        exit : int, model exit for model results
        std : float, standard deviation for noise or blur if applicable
    
    Returns:
    --------
        df : pandas DataFrame with columns [targets, predictions, experiment, exit, std
    '''
    df = pd.DataFrame()
    n = len(targets)
    df['model'] = np.repeat(model, n)
    if experiment == '':
        df['experiment'] = 'color'
    else:
        df['experiment'] = experiment
    df['exit'] = exit
    if std != None:
        df['std'] = std
    df['targets'] = targets
    df['predictions'] = preds
    return df

def main(args):
    #Set up references to appropriate data folders. Not that '' = color
    sub_exps = ['', 'grayscale', 'grayscale,random_gauss_noise', 'random_gauss_blur']
    exp_folders = {'':'ColorGraySplit', 
                'grayscale': 'ColorGraySplit', 
                'grayscale,random_gauss_noise': 'NoiseSplit_gray_contrast0.2', 
                'random_gauss_blur':'BlurSplit_color'} 

    noise_std = [0, 0.04, 0.16]
    blur_std = [0, 3, 9]

    output_reps = {}

    subfolder_indices = {}

    np.random.seed(42)

    labels = ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck','car', 'bird', 'dog' ]

    #Setup if we're calculating accuracy by category
    if args.get_acc_by_cat:
        color_cat_indices = {}
        acc_by_cat = {}

    for exp in sub_exps:
        if exp in(['', 'grayscale']):
            subfolder_indices[exp] = []
            subfolder_indices[exp].append(0)
            total_files = 0
            for i in range(5):
                if exp == '':
                    num_files = len([f for f in os.listdir(os.path.join(args.human_data_dir, exp_folders[exp], str(i))) if str.find(f, '_c_')>-1])

                    if args.get_acc_by_cat:
                        color_cat_indices[i] = {}
                        color_files = [f for f in os.listdir(os.path.join(args.human_data_dir, exp_folders[exp], str(i))) if str.find(f, '_c_')>-1]
                        
                        for label in labels:
                            color_cat_indices[i][label] = [i for i, f in enumerate(color_files) if str.find(f, label)>-1]
                elif exp == 'grayscale':
                    num_files = len([f for f in os.listdir(os.path.join(args.human_data_dir, exp_folders[exp], str(i))) if str.find(f, '_g_')>-1])
                subfolder_indices[exp].append(int(total_files+num_files))
                total_files +=num_files
        elif exp == 'grayscale,random_gauss_noise':
            subfolder_indices[exp] = {}
            for std in noise_std:
                subfolder_indices[exp][std] = []
                subfolder_indices[exp][std].append(0)
                total_files = 0
                for i in range(5):
                    num_files = len([f for f in os.listdir(os.path.join(args.human_data_dir, exp_folders[exp], str(i))) if str.find(f, f'_{std}_')>-1])
                    subfolder_indices[exp][std].append(int(total_files+num_files))
                    total_files +=num_files
        elif exp == 'random_gauss_blur':
            subfolder_indices[exp] = {}
            for std in blur_std:
                subfolder_indices[exp][std] = []
                subfolder_indices[exp][std].append(0)
                total_files = 0
                for i in range(5):
                    num_files = len([f for f in os.listdir(os.path.join(args.human_data_dir, exp_folders[exp], str(i))) if str.find(f, f'_{int(std)}_')>-1])
                    subfolder_indices[exp][std].append(int(total_files+num_files))
                    total_files +=num_files        

    exits = [0,2,4,6,8]
    accuracies = {}


    output_df = pd.DataFrame(columns = ['model', 'experiment', 'exit', 'targets', 'predictions'])
    experiments_dir = os.path.join(args.experiments_root, args.dataset_name, 'experiments')
    for mod in args.models:
        accuracies[mod] = {}
        output_reps[mod] = {}
        acc_by_cat[mod] = {}
        for exp in sub_exps: 
            if exp in ['', 'grayscale']:
                accuracies[mod][exp] = []
                if exp == '':
                    output_rep_file = os.path.join(experiments_dir, f'td{mod},lr_0.01,wd_0.0005,seed_42', 'outputs', 'output_representations__test_human__OSD.pt')
                else:
                    output_rep_file = os.path.join(experiments_dir, f'td{mod},lr_0.01,wd_0.0005,seed_42,{exp}', 'outputs', 'output_representations__test_human__OSD.pt')

                print(f"Reading output reps from {output_rep_file}")
                output_reps[mod][exp] = torch.load(output_rep_file)
                preds = output_reps[mod][exp]['predictions']
                targets = output_reps[mod][exp]['target']
                accuracies[mod][exp] = {}
                accuracies[mod][exp]['sample1'] = {'accuracies': [],
                                                    'sample_sizes': []}
                accuracies[mod][exp]['sample2'] = {'accuracies': [],
                                                    'sample_sizes': []}

                for idx, e in enumerate(exits):
                    all_sub_indices = np.arange(subfolder_indices[exp][idx], subfolder_indices[exp][idx+1])
                    preds_all = preds[e, all_sub_indices]
                    targets_all = targets[all_sub_indices]
                    
                    add_df = create_df(targets_all, preds_all, mod, exp, e)
                    output_df = pd.concat([output_df, add_df], axis = 0)

                    n_indices = len(all_sub_indices)
                    n_samp1 = np.int(np.round(n_indices/2))

                    samp1_indices = np.random.choice(all_sub_indices, n_samp1)
                    samp2_indices = [i for i in all_sub_indices if i not in (samp1_indices)]
                    
                    preds_slice1 = preds[e, samp1_indices]
                    targets_slice1 = targets[samp1_indices]
                    acc1 = torch.sum(preds_slice1 == targets_slice1)/len(targets_slice1)
                    accuracies[mod][exp]['sample1']['accuracies'].append(acc1.item())
                    accuracies[mod][exp]['sample1']['sample_sizes'].append(n_samp1)

                    preds_slice2 = preds[e, samp2_indices]
                    targets_slice2 = targets[samp2_indices]
                    acc2 = torch.sum(preds_slice2 == targets_slice2)/len(targets_slice2)
                    accuracies[mod][exp]['sample2']['accuracies'].append(acc2.item())
                    accuracies[mod][exp]['sample2']['sample_sizes'].append(np.int(n_indices -n_samp1))

                if exp == '' and args.get_acc_by_cat:
                    for label in labels:
                        acc_by_cat[mod][label] = []

                        for idx, e in enumerate(exits):
                            label_idx = color_cat_indices[idx][label]
                            preds_label = preds[e, label_idx]
                            targets_label = targets[label_idx]
                            acc_label = torch.sum(preds_label == targets_label)/len(targets_label)
                            acc_by_cat[mod][label].append(acc_label.item())


            elif exp == 'grayscale,random_gauss_noise':
                output_reps[mod][exp] = {}
                accuracies[mod][exp] = {}
                for std in noise_std:
                    output_rep_file = os.path.join(experiments_dir, f'td{mod},lr_0.01,wd_0.0005,seed_42,{exp}', 'outputs', f'output_representations__test_human__OSD__gauss_noise{std}.pt')
                    print(f"Reading output reps from {output_rep_file}")
                    output_reps[mod][exp][std] = torch.load(output_rep_file)

                    preds = output_reps[mod][exp][std]['predictions']
                    targets = output_reps[mod][exp][std]['target']
                    accuracies[mod][exp][std] = {}
                    accuracies[mod][exp][std]['sample1'] = {'accuracies': [],
                                                            'sample_sizes': []}
                    accuracies[mod][exp][std]['sample2'] = {'accuracies': [],
                                                            'sample_sizes': []}
                    for idx, e in enumerate(exits):
                        all_sub_indices = np.arange(subfolder_indices[exp][std][idx], subfolder_indices[exp][std][idx+1])
                        preds_all = preds[e, all_sub_indices]
                        targets_all = targets[all_sub_indices]
                        
                        add_df = create_df(targets_all, preds_all, mod, exp, e, std)
                        output_df = pd.concat([output_df, add_df], axis = 0)

                        n_indices = len(all_sub_indices)
                        n_samp1 = np.int(np.round(n_indices/2))

                        samp1_indices = np.random.choice(all_sub_indices, n_samp1)
                        samp2_indices = [i for i in all_sub_indices if i not in (samp1_indices)]

                        preds_slice1 = preds[e, samp1_indices]
                        targets_slice1 = targets[samp1_indices]
                        acc1 = torch.sum(preds_slice1 == targets_slice1)/len(targets_slice1)
                        accuracies[mod][exp][std]['sample1']['accuracies'].append(acc1.item())
                        accuracies[mod][exp][std]['sample1']['sample_sizes'].append(n_samp1)

                        preds_slice2 = preds[e, samp2_indices]
                        targets_slice2 = targets[samp2_indices]
                        acc2 = torch.sum(preds_slice2 == targets_slice2)/len(targets_slice2)
                        accuracies[mod][exp][std]['sample2']['accuracies'].append(acc2.item())
                        accuracies[mod][exp][std]['sample2']['sample_sizes'].append(np.int(n_indices -n_samp1))

            
            elif exp == 'random_gauss_blur':
                output_reps[mod][exp] = {}
                accuracies[mod][exp] = {}
                for std in blur_std:
                    output_rep_file = os.path.join(experiments_dir, f'td{mod},lr_0.01,wd_0.0005,seed_42,{exp}', 'outputs', f'output_representations__test_human__OSD__blur{std}.pt')
                    print(f"Reading output reps from {output_rep_file}")
                    output_reps[mod][exp][std] = torch.load(output_rep_file)

                    preds = output_reps[mod][exp][std]['predictions']
                    targets = output_reps[mod][exp][std]['target']

                    accuracies[mod][exp][std] = {}

                    accuracies[mod][exp][std]['sample1'] = {'accuracies': [],
                                                            'sample_sizes': []}
                    accuracies[mod][exp][std]['sample2'] = {'accuracies': [],
                                                            'sample_sizes': []}

                    for idx, e in enumerate(exits):
                        all_sub_indices = np.arange(subfolder_indices[exp][std][idx], subfolder_indices[exp][std][idx+1])
                        preds_all = preds[e, all_sub_indices]
                        targets_all = targets[all_sub_indices]
                        
                        add_df = create_df(targets_all, preds_all, mod, exp, e, std)
                        output_df = pd.concat([output_df, add_df], axis = 0)

                        n_indices = len(all_sub_indices)
                        n_samp1 = np.int(np.round(n_indices/2))

                        samp1_indices = np.random.choice(all_sub_indices, n_samp1)
                        samp2_indices = [i for i in all_sub_indices if i not in (samp1_indices)]

                        preds_slice1 = preds[e, samp1_indices]
                        targets_slice1 = targets[samp1_indices]
                        acc1 = torch.sum(preds_slice1 == targets_slice1)/len(targets_slice1)
                        accuracies[mod][exp][std]['sample1']['accuracies'].append(acc1.item())
                        accuracies[mod][exp][std]['sample1']['sample_sizes'].append(n_samp1)

                        preds_slice2 = preds[e, samp2_indices]
                        targets_slice2 = targets[samp2_indices]
                        acc2 = torch.sum(preds_slice2 == targets_slice2)/len(targets_slice2)
                        accuracies[mod][exp][std]['sample2']['accuracies'].append(acc2.item())
                        accuracies[mod][exp][std]['sample2']['sample_sizes'].append(np.int(n_indices -n_samp1))

    with open(os.path.join(args.output_dir, 'accuracies_diff_exits_error_bars.json'), 'w') as outfile:
        json.dump(accuracies, outfile)


    with open(os.path.join(args.output_dir, 'accuracies_by_cat.json'), 'w') as outfile:
        json.dump(acc_by_cat, outfile)


    output_df.to_csv(os.path.join(args.output_dir, 'targets_predictions_by_exp.csv'), index = False)

if __name__ == "__main__":
  args = setup_args()
  main(args)



        

