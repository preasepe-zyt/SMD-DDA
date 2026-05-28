from typing import List

import torch
from tqdm import tqdm
import numpy as np

from .args import PredictArgs, TrainArgs
from .model import MoleculeModel
from .make_predictions import make_predictions

def pre_gene(dataset):
   arguments = [
    '--checkpoint_path', './pre_gene/model.pt',
   '--test_path', dataset,
  '--preds_path', './pre_gene/CTP_preds.csv']

   args = PredictArgs().parse_args(arguments)
   data = make_predictions(args)
   data_tensor = torch.tensor(data, device=args.device)
   return data_tensor

if __name__ == "__main__":
   dataset = '/home/zyfone/hard-disk/zyt/drug-disease/dataset/Gdataset_drugs.csv'
   dat = pre_gene(dataset)



