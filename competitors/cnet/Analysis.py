"""Script for generating SAT curve analysis plots for CNet"""
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import torch
from collections import defaultdict
from datasets.dataset_handler import DataHandler
from matplotlib.lines import Line2D
from modules import utils
from scipy import interpolate
import pickle

#DATASET_ROOT = "/scratch/work/public/imagenet" #"/scratch/sbp354/cascade_output/datasets"  #"/scratch/work/public/imagenet"
#EXPERIMENT_ROOT = "/scratch/sbp354/cascade_output/experiments" 
#SPLIT_IDX_ROOT = "/scratch/sbp354/cascade_output/split_idx"
#TEST_DATASET_ROOT='/scratch/sbp354/SAT_human_data'
#output_root = '/scratch/sbp354/CascadedNets/output_files'

#DATASET_NAME = "ImageNet2012_16classes_rebalanced"
#MODEL_KEY = 'resnet18'
#DATASET_KEY = 'test_human'

#COLOR = True
#GRAYSCALE = False
#GAUSS_NOISE = False
#BLUR = False
#BLUR_RANGE = True
#BLUR_RANGE_FM = ''

plot_category_accuracies = True

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--random_seed", type=int, default=42,
                      help="random seed")
  
  # Paths
  parser.add_argument("--experiment_root", type=str,required = True,
                      help="Local output dir")
  # Dataset
  parser.add_argument("--dataset_root", type=str, required = True,
                      help="Dataset root")
  parser.add_argument("--dataset_name", type=str, default='ImageNet2012_16classes_rebalanced',
                        help="Dataset name: CIFAR10, CIFAR100, TinyImageNet, ImageNet2012_16classes_rebalanced")
  parser.add_argument("--split_idxs_root", type=str, required= True,
                      help="Split idxs root")
  parser.add_argument("--test_dataset_root", type=str, required = True,
                      help="Directory where test dataset is stored")
  parser.add_argument("--output_root", type =str, required = True, 
                      help = "Parent directory where SAT curves will be output")
  parser.add_argument("--dataset_key", type=str,
                      help="Indicates which testing dataset to use in analysis: test_human, test")
  parser.add_argument("--val_split", type=float, default=0.1,
                      help="Validation set split: 0.1 default")
  parser.add_argument("--augmentation_noise_type", type=str, 
                      default='occlusion',
                      help="Augmentation noise type: occlusion")
  parser.add_argument("--batch_size", type=int, default=128,
                      help="batch_size")
  parser.add_argument("--num_workers", type=int, default=2,
                      help="num_workers")
  parser.add_argument('--drop_last', action='store_true', default=False,
                      help='Drop last batch remainder')
  
  # Model
  parser.add_argument("--model_key", type=str, default='resnet18',
                      help="Model: resnet18, resnet34, ..., densenet_cifar")
  parser.add_argument("--train_mode", type=str, 
                      default='baseline',
                      help="Train mode: baseline, ic_only, sdn")
  parser.add_argument('--bn_time_affine', action='store_true', default=False,
                      help='Use temporal affine transforms in BatchNorm')
  parser.add_argument('--bn_time_stats', action='store_true', default=False,
                      help='Use temporal stats in BatchNorm')
  parser.add_argument("--tdl_mode", type=str, 
                      default='OSD',
                      help="TDL mode: OSD, EWS, noise")
  parser.add_argument("--tdl_alpha", type=float, default=0.0,
                      help="TDL alpha for EWS temporal kernel")
  parser.add_argument("--noise_var", type=float, default=0.0,
                      help="Noise variance on noise temporal kernel")
  parser.add_argument("--lambda_val", type=float, default=1.0,
                      help="TD lambda value")
  parser.add_argument('--cascaded', action='store_true', default=False,
                      help='Cascaded net')
  parser.add_argument("--cascaded_scheme", type=str, default='parallel',
                      help="cascaded_scheme: serial, parallel")
  parser.add_argument("--init_tau", type=float, default=0.01,
                      help="Initial tau valu")
  parser.add_argument("--target_IC_inference_costs", nargs="+", type=float, 
                      default=[0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
                      help="target_IC_inference_costs")
  parser.add_argument('--tau_weighted_loss', action='store_true', default=False,
                      help='Use tau weights on IC losses')
  parser.add_argument('--color', action = argparse.BooleanOptionalAction,
                      help='Boolean for if model results are color')
  parser.add_argument('--grayscale',  action = argparse.BooleanOptionalAction,
                      help='Boolean for if model results are grayscale')
  parser.add_argument('--gauss_noise',action = argparse.BooleanOptionalAction,
                      help='Boolean for if model results are using gaussian noise')
  parser.add_argument('--blur', action = argparse.BooleanOptionalAction,
                      help='Boolean for if model results are using gaussian blur')
  parser.add_argument('--blur_range', action = argparse.BooleanOptionalAction, default = False,
                      help='Boolean for if model results are using gaussian blur range')      
  parser.add_argument('--blur_range_fm', type = str, default = '')               
  
  # Optimizer
  parser.add_argument("--learning_rate", type=float, default=0.1,
                      help="learning rate")
  parser.add_argument("--momentum", type=float, default=0.9,
                      help="momentum")
  parser.add_argument("--weight_decay", type=float, default=0.0005,
                      help="weight_decay")
  parser.add_argument('--nesterov', action='store_true', default=False,
                      help='Nesterov for SGD')
  parser.add_argument('--normalize_loss', action='store_true', default=False,
                      help='Normalize temporal loss')
  
  # LR scheduler
  parser.add_argument("--lr_milestones", nargs="+", type=float, 
                      default=[60, 120, 150],
                      help="lr_milestones")
  parser.add_argument("--lr_schedule_gamma", type=float, default=0.2,
                      help="lr_schedule_gamma")
  
  # Other
  parser.add_argument('--use_cpu', action='store_true', default=False,
                      help='Use cpu')
  parser.add_argument("--device", type=int, default=0,
                      help="GPU device num")
  parser.add_argument("--n_epochs", type=int, default=150,
                      help="Number of epochs to train")
  parser.add_argument("--eval_freq", type=int, default=10,
                      help="eval_freq")
  parser.add_argument("--save_freq", type=int, default=5,
                      help="save_freq")
  parser.add_argument('--keep_logits', action='store_true', default=False,
                      help='Keep logits')
  parser.add_argument('--debug', action='store_true', default=False,
                      help='Debug mode')
  
  args = parser.parse_args()
  
  # Flag check
  if args.tdl_mode == 'EWS':
    assert args.tdl_alpha is not None, 'tdl_alpha not set'
  elif args.tdl_mode == 'noise':
    assert args.noise_var is not None, 'noise_var not set'
    
  return args

args = setup_args()

def main(args):
  # Set required flags|
  if "imagenet2012" in args.dataset_name.lower():
    args.experiment_name = f"{args.model_key}_{args.dataset_name}"
  elif "cifar" in args.dataset_name.lower():
    args.experiment_name = f"{args.model_key}_{args.dataset_name}"
  elif "stl" in args.dataset_name.lower():
    args.experiment_name = f"{args.model_key}_{args.dataset_name}"
  else:
    print("TinyImageNet not implemented yet!")
    
  args.val_split = 0.1
  args.test_split = 0.1
  args.tdl_mode = 'OSD'  # OSD, EWS
  args.tau_weighted_loss = True
  args.random_seed = 42
  args.grayscale = False


  #Set up output directory
  output_folder = os.path.join(args.output_root, args.dataset_name)
  if args.dataset_name == 'ImageNet2012_16classes_rebalanced':
    output_folder = os.path.join(output_folder, args.dataset_key)

  if args.grayscale:
      output_folder = os.path.join(output_folder, 'grayscale')
      if args.blur:
          if not args.blur_range:
            output_folder = os.path.join(output_folder, 'blur')
          else:
            output_folder = os.path.join(output_folder, 'blur_range')
      elif args.gauss_noise:
          output_folder = os.path.join(output_folder, 'gauss_noise')
  else:
      output_folder = os.path.join(output_folder, 'color')
      if args.blur:
          output_folder = os.path.join(output_folder, 'blur')
      elif args.gauss_noise:
          output_folder = os.path.join(output_folder, 'gauss_noise')

  if not os.path.exists(output_folder):
      os.makedirs(output_folder)

  #Model Label Lookup
  MODEL_LBL_LOOKUP = {
      "cascaded__serial": "SerialTD",
      "cascaded__parallel": "CascadedTD",
      "cascaded__serial__multiple_fcs": "SerialTD-MultiHead",  # (SDN)
      "cascaded__parallel__multiple_fcs": "CascadedTD-MultiHead",
      "cascaded_seq__serial": "SerialCE",
      "cascaded_seq__parallel": "CascadedCE",
  }

  #Plotting colors
  colors_src = {
      "CascadedTDColor": np.array([182,54,121]) / 255.0,  # CascadedTDColor,
      "CascadedCEColor": np.array([127,38,110]) / 255.0,  # CascadedCEColor,
      "SerialTDColor": np.array([77,184,255]) / 255.0,  # SerialTDColor,
      "SerialCEColor": np.array([54,129,179]) / 255.0,  # SerialCEColor,
  }

  model_colors = {
      "cascaded__serial": colors_src["SerialTDColor"],
      "cascaded__parallel": colors_src["CascadedTDColor"],
      "cascaded__serial__multiple_fcs": colors_src["SerialTDColor"],  # (SDN)
      "cascaded__parallel__multiple_fcs": colors_src["CascadedTDColor"],
      "cascaded_seq__serial": colors_src["SerialCEColor"],
      "cascaded_seq__parallel": colors_src["CascadedCEColor"],
  }

  #Setting up data handler
  # Data Handler
  data_dict = {
      "dataset_name": args.dataset_name,
      "experiment_root":args.experiment_root,
      "test_dataset_root":args.test_dataset_root,
      "data_root": args.dataset_root,
      "val_split": args.val_split,
      "test_split": args.test_split,
      "split_idxs_root": args.split_idxs_root,
      "noise_type": args.augmentation_noise_type,
      "load_previous_splits": True,
      "grayscale": args.grayscale,
      "gauss_noise":args.gauss_noise,
      "gauss_noise_std":0,
      "blur":args.blur,
      "blur_std":0.0,
      "imagenet_params": {
        #"target_classes": ["terrier"],
        "max_classes": 1000,
      }
  }
  data_handler = DataHandler(**data_dict)
  test_loader = data_handler.build_loader(args.dataset_key, args)

  #Load experiment data

  # Set experiment root
  exp_root = os.path.join(args.experiment_root,
                          args.experiment_name,
                          'experiments')

  # Find exp paths
  if args.grayscale:
    if args.blur:
      exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}__*.pt'))\
                  if (str.find(i,'grayscale')>-1) & (str.find(i,'random_gauss_blur')>-1)]

    elif args.gauss_noise:
      exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}_*.pt'))\
                  if (str.find(i,'grayscale')>-1) & (str.find(i,'random_gauss_noise')>-1)]
      if args.dataset_key == 'test_human':
        exp_paths = [i for i in exp_paths if (str.find(i, 'noise0.pt') >-1) | (str.find(i, 'noise0.04.pt') >-1)| (str.find(i, 'noise0.16.pt') >-1)]
  
    else:
      exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}.pt'))\
                  if (str.find(i,'grayscale')>-1) & (str.find(i,'blur')==-1) & (str.find(i,'noise')==-1)]
   
  else:
    if args.blur:
      if not args.blur_range:
        exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}*.pt'))\
                  if (str.find(i,'grayscale')==-1) & (str.find(i,'random_gauss_blur')>-1)]
      else:
        exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}*.pt'))\
                  if (str.find(i,'grayscale')==-1) & (str.find(i,'random_gauss_blur')>-1)]
        exp_paths = [i for i in exp_paths if (str.find(i, 'step'))]
      
      if args.dataset_key == 'test_human':
        exp_paths = [i for i in exp_paths if (str.find(i, 'blur0.pt') >-1) | (str.find(i, 'blur0.3.pt') >-1)| (str.find(i, 'blur0.9.pt') >-1)]
    elif args.gauss_noise:
      exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}*.pt'))\
                  if (str.find(i,'grayscale')==-1) & (str.find(i,'random_gauss_noise')>-1)]
      if args.dataset_key == 'test_human':
        exp_paths = [i for i in exp_paths if i[-6:]=='OSD.pt']
    else:
      exp_paths = [i for i in np.sort(glob.glob(f'{exp_root}/*/outputs/*__{args.dataset_key}__{args.tdl_mode}.pt'))\
                  if (str.find(i,'grayscale')==-1) & (str.find(i,'blur')==-1) & (str.find(i,'noise')==-1)]

  print(f"# experiment paths: {len(exp_paths)}")

  df_dict = defaultdict(list)
  outrep_id = 0
  outreps_dict = {}
  ic_costs_lookup = {}
  exp_path_lookup = {}
  for i, exp_path in enumerate(exp_paths):
    outrep_id = f'rep_id_{i}'
    outrep = torch.load(exp_path)
    
    basename = [ele for ele in exp_path.split(os.path.sep) if 'seed_' in ele][0]
    keys = basename.split(',')
    if keys[0].startswith('std') or keys[0].startswith('cascaded_seq'):
      model_key, lr, weight_decay, seed = keys
      td_key = 'std'
    else:
      td_key, scheme_key, lr, weight_decay, seed = keys[:5]
      model_key = f'cascaded__{scheme_key}'
      if str.find(exp_path, '0.0_0.005_0.01')> -1:
        model_key = model_key + '_0.0_0.005_0.01'
      other_keys = keys[5:]
      multiple_fcs = 'multiple_fcs' in other_keys
      tau_weighted = 'tau_weighted' in other_keys
      pretrained_weights = 'pretrained_weights' in other_keys
      if multiple_fcs:
        model_key = f'{model_key}__multiple_fcs'
      if tau_weighted:
        model_key = f'{model_key}__tau_weighted'
      if pretrained_weights:
        model_key = f'{model_key}__pretrained_weights'

    if model_key != 'std':
      exp_root = os.path.dirname(os.path.dirname(exp_path))
      IC_cost_path = os.path.join(exp_root, 'ic_costs.pt')
      if os.path.exists(IC_cost_path):
        IC_costs = torch.load(IC_cost_path)
      else:
        IC_costs = None
    else:
      IC_costs = None
      
    lr = float(lr.split("_")[1])
    weight_decay = float(weight_decay.split("_")[1])
    
    if args.blur:
      if str.find(exp_path, "OSD__blur")==-1:
        model_key = model_key
      else:
        blur_std = exp_path[str.find(exp_path, "OSD__blur")+9:-3]
        model_key = f"{model_key}_{blur_std}"
    elif args.gauss_noise:
      if str.find(exp_path, "OSD__gauss_noise")==-1:
        model_key = model_key
      else:
        gauss_noise_std = exp_path[str.find(exp_path, "OSD__gauss_noise")+16:-3]
        model_key = f"{model_key}_{gauss_noise_std}"
      
    df_dict['model'].append(model_key)
    df_dict['td_lambda'].append(td_key)
    df_dict['lr'].append(lr)
    df_dict['weight_decay'].append(weight_decay)
    df_dict['seed'].append(seed)
    df_dict['outrep_id'].append(outrep_id)

    outreps_dict[outrep_id] = outrep
    ic_costs_lookup[outrep_id] = IC_costs
    exp_path_lookup[outrep_id] = exp_path
  analysis_df = pd.DataFrame(df_dict)
  analysis_df.to_csv(os.path.join(output_folder, 'analysis_df.csv'))
  #Build aggregate stats
  df_dict = defaultdict(list)
  analysis_df = analysis_df.sort_values('model')

  accs_dict = {}
  for i, df_i in analysis_df.iterrows():
    outrep_i = outreps_dict[df_i.outrep_id]
    accs = outrep_i['correct'].float().mean(dim=1)
    accs_dict[f"{df_i['model']}_{df_i['td_lambda']}"] = accs

    #category specific accuracies:
    fig = plt.figure()
    t_steps = range(9)
    output_folder_lbl_accuracy = os.path.join(output_folder, 'accuracy_by_label')
    if not os.path.exists(output_folder_lbl_accuracy):
      os.mkdir(output_folder_lbl_accuracy)

    for t in outrep_i['target'].unique():
      t_index = (outrep_i['target']==t).nonzero(as_tuple=True)[0]
      t_correct = torch.index_select(outrep_i['correct'], 1, t_index)
      t_accs = t_correct.float().mean(dim=1)


      plt.plot(t_steps, t_accs, label = f'Label = {t}')
      plt.title(f"SAT curve {df_i['model']} {df_i['td_lambda']} by target label")
      plt.legend(loc = 'lower right')
      output_name = os.path.join(output_folder_lbl_accuracy, f"SAT_plot_{df_i['model']}_{df_i['td_lambda']}.png")
      plt.savefig(output_name)
  
    

    for i, acc in enumerate(accs):
      df_dict['acc'].append(acc.item() * 100)
      if len(accs) == 1:
        i = -1
      df_dict['ic'].append(i)
      for k in list(df_i.index):
        df_dict[k].append(df_i[k])

  pkl_file = open(os.path.join(output_folder, 'accuracy_dictionary.pkl'), 'wb')
  pickle.dump(accs_dict, pkl_file)
  pkl_file.close()

  accs_df = pd.DataFrame(df_dict)

  accs_df = accs_df.sort_values(['outrep_id', 'ic'])

  print(args.blur)
  print(args.gauss_noise)
  if ((not args.blur) & (not args.gauss_noise)) | ((args.blur |args.gauss_noise) & (args.dataset_key=='test_human')):
    fig = plt.figure(figsize = (15,10))
    plt.rc('font', size=16) 
    for k in accs_dict.keys():
      plt.plot(t_steps, accs_dict[k], label = k)
      plt.legend(bbox_to_anchor=(1.6, 0.5), loc = 'lower right')
      plt.tight_layout()

    if args.grayscale:
      plt.title(f"SAT Curve Grayscale")
      save_path = os.path.join(output_folder, 'SAT_curve_grayscale.png')
      plt.savefig(save_path, bbox_inches = 'tight')
      print(f"Outputting SAT Curve for grayscale images to :{save_path}")
    else:
      plt.title(f"SAT Curve Color")
      save_path = os.path.join(output_folder, 'SAT_curve_color.png')
      plt.savefig(save_path, bbox_inches = 'tight')
      print(f"Outputting SAT Curve for color images to :{save_path}")
  else:
    parent_model_keys = set([i[:str.rfind(i, "_", 0, len(i)-8)] for i in accs_dict.keys()])
    
    for k in parent_model_keys:
      for t in [0.0, 0.5, 1.0]:
        if (args.blur) and (args.color) and (args.dataset_name == 'ImageNet2012_16classes_rebalanced'):
          child_model_keys = [i for i in accs_dict.keys() if (str.find(i, k)>-1) & (str.find(i, f'td({t})')>-1) & (str.find(i, '.25')==-1) & (str.find(i, '.75')==-1)][:14]
        else:
          child_model_keys = [i for i in accs_dict.keys() if (str.find(i, k)>-1) & (str.find(i, f'td({t})')>-1)][:8]
    
        
        if len(child_model_keys)>0:
          fig = plt.figure()
        
          if args.blur:
            for i,m in enumerate(child_model_keys):
              plt.plot(t_steps, accs_dict[m], label = f'blur std {m[str.rfind(m, "_", 0, len(m)-8):-8]}')
              plt.legend(loc = 'best')
            if args.grayscale:
              plt.title(f"SAT Curve Grayscale Blur {k}_td({t})")
              plt.savefig(os.path.join(output_folder, f'SAT_curve_grayscale_blur_{k}_td({t}).png'),bbox_inches = 'tight')
            else:
              if args.blur_range:
                plt.title(f"SAT Curve Color Blur {args.blur_range_fm} {k}_td({t})")
                plt.savefig(os.path.join(output_folder, f'SAT_curve_color_blur_{args.blur_range_fm}_{k}_td({t}).png'),bbox_inches = 'tight')
          elif args.gauss_noise:
            for i,m in enumerate(child_model_keys):
              plt.plot(t_steps, accs_dict[m], label = f'noise std {m[str.rfind(m, "_", 0, len(m)-8):-8]}')
              plt.legend(loc = 'best')
            if args.grayscale:
              plt.title(f"SAT Curve Grayscale Gaussian Noise {k}_td({t})")
              plt.savefig(os.path.join(output_folder, f'SAT_curve_grayscale_gauss_noise_{k}_td({t}).png'))
            else:
              plt.title(f"SAT Curve Color Gaussian Noise {k}_td({t})")
              plt.savefig(os.path.join(output_folder, f'SAT_curve_color_gauss_noise_{k}_td({t}).png'))
          plt.close()

if __name__ == "__main__":
  args = setup_args()
  main(args)

