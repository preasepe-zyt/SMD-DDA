import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from model import MoleculeModel
import numpy as np

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
def pcc_loss(pred, target):
    vx = pred - pred.mean(dim=1, keepdim=True)
    vy = target - target.mean(dim=1, keepdim=True)
    corr = (vx * vy).mean(dim=1) / (vx.std(dim=1) * vy.std(dim=1) + 1e-8)
    return 1 - corr.mean()

    
def train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """


    model.train()
    loss_sum = iter_count = 0
    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        targets = torch.tensor(
    [[0 if x is None else x for x in tb] for tb in batch.genes],
    dtype=torch.float32,
    device=args.device
)

        # Run model
        model.zero_grad()
        preds = model(batch.to(args.device))

        # Move tensors to correct device
        torch_device = preds.device
        targets = targets.to(torch_device)
        loss = loss_func(preds, targets)
        num_elements = targets.numel()
        loss = loss.sum() / num_elements
        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0
            rmse_value = rmse(preds.detach().cpu().numpy(),
                  targets.detach().cpu().numpy())
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            print(f'Loss = {loss_avg:.4e}, {lrs_str}, RMSE_train = {rmse_value}')



    return n_iter