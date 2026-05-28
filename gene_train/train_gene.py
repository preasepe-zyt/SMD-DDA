import os
import chemprop
from cross_validate import cross_validate
from run_training import run_training
import torch
arguments = [
    '--data_path', 'ctp_phase1+2.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'ctp_phase1+2-re'
]

device = torch.device("cuda:0")  # 这里必须是 0，因为映射后只剩一张 GPU
args = chemprop.args.TrainArgs().parse_args(arguments)
args.device = device
args.save_smiles_splits = True
args.save_preds = True
args.epochs = 100
args.batch_size = 64
args.final_lr =  0.0003
args.loss_function = 'mse_dir'
args.conformal_alpha = 1
args.num_folds = 2
mean_score, std_score = cross_validate(args=args)