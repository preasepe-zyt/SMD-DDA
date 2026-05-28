import os
import collections
import time
import multiprocessing
from copy import deepcopy
from argparse import Namespace
from typing import List, Optional, Dict
import numpy as np
import gc
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter
from cliff.utils.utils import save_checkpoint, load_checkpoint, pairwise_ranking_loss, get_deg
from cliff.utils.metrics import get_metric_func
from cliff.dataset import MoleculeDataset
from cliff.models.GNN import GNN
from sklearn.model_selection import train_test_split
import pandas as pd

def run_training(args: Namespace,
                 data_train: List[Data], 
                 data_val: List[Data],
                 ) -> Dict[str, float]:
    """
    :param model: Model to train.
    :param data_train: Training data.
    :param data_val: Validation data.
    :return: Dictionary of best validation scores for each metric.
    """
      
    model = GNN()
    train_loader = DataLoader(data_train, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size = args.batch_size, shuffle=False)

    loss_func = torch.nn.MSELoss() if args.task == 'regression' else torch.nn.BCEWithLogitsLoss()
    
    # Create dictionary of metric functions
    metric_funcs = {metric: get_metric_func(metric=metric) for metric in args.metric}
    
    # Track best scores for each metric
    best_scores = {}
    best_epochs = {}
    for metric in args.metric:
        best_scores[metric] = float('inf') if metric in ['rmse', 'mse', 'mae'] else -float('inf')
        best_epochs[metric] = 0
    
    # Also track best validation loss for early stopping
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use validation loss for LR scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=args.factor, patience=args.patience, min_lr=args.min_lr)

    losses = collections.defaultdict(list)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):

        s_time = time.time()
        train_losses, y_true, output, mol_f = train(args, epoch, model, train_loader, loss_func, optimizer, device)
        if epoch > 0:
           #mol_f_cpu = mol_f.detach().cpu()
           #torch.save(mol_f_cpu, f"{args.dataset}_mol_f.pt")
           y_true_np = y_true.detach().cpu().numpy().reshape(-1)

           df_y = pd.DataFrame({
    "y_true": y_true_np
})
           df_y.to_csv(f"{args.dataset}_y_true.csv", index=False)
        t_time = time.time() - s_time
        s_time = time.time()
        
        # Get all metrics
        val_scores, val_loss = evaluate(args, model, val_loader, loss_func, metric_funcs, device)
        
        v_time = time.time() - s_time
        # Use validation loss for scheduler
        scheduler.step(val_loss)

        losses['train_pred'].append(train_losses['pred'][0])
        losses['val'].append(val_loss)

        print('Epoch: {:04d}'.format(epoch),
                'train_pred_loss: {:.6f}'.format(train_losses['pred'][0]),
                'val_pred_loss: {:.6f}'.format(val_loss),
                'cur_lr: {:.5f}'.format(optimizer.param_groups[0]['lr']),
                't_time: {:.4f}s'.format(t_time),
                'v_time: {:.4f}s'.format(v_time))
                
        # Print all metrics
        for metric, score in val_scores.items():
            print('{:.4s}_val: {:.4f}'.format(metric, score))
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            # Save a model based on validation loss
            if args.save_checkpoints:
                os.makedirs(os.path.join(args.model_dir, args.dataset), exist_ok=True)
                checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                             f'{args.dataset}_{args.loss}_model_{args.seed}_val_loss.pt')
                save_checkpoint(checkpoint_path, model, args)
                #print(f'Saved best validation loss model at epoch {epoch} at {checkpoint_path}')
                
        # Check for improvement in each metric and save corresponding model
        for metric, score in val_scores.items():
            minimize = metric in ['rmse', 'mse', 'mae']
            if (minimize and score < best_scores[metric]) or (not minimize and score > best_scores[metric]):
                best_scores[metric], best_epochs[metric] = score, epoch
                if args.save_checkpoints:
                    os.makedirs(os.path.join(args.model_dir, args.dataset), exist_ok=True)
                    checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                                 f'{args.dataset}_{args.loss}_model_{args.seed}_{metric}.pt')
                    save_checkpoint(checkpoint_path, model, args)
                    #print(f'Saved best {metric} model at epoch {epoch} at {checkpoint_path}')
                    
            
    # Print best results for each metric
    print('Best validation scores:')
    print('val_loss: {:.4f} at epoch {:04d}'.format(best_val_loss, best_val_loss_epoch))
    for metric, score in best_scores.items():
        print('{:.4s}_val: {:.4f} at epoch {:04d}'.format(metric, score, best_epochs[metric]))
    
    del model
    torch.cuda.empty_cache()
    return best_scores[args.metric[0]]
        

def train(args, epoch, model, train_loader, loss_func, optimizer, device):
    """
    Trains a model for an epoch.
    """
    model.train()
    losses = collections.defaultdict(list)
    total_loss, pred_loss = 0.0, 0.0
    len_dataloader = len(train_loader)
    all_target = []
    all_output = []
    all_mol_f = []
    for i, data in enumerate(train_loader):
        data.to(device)
        target = data.target.reshape(-1, 1).to(device)
        output, mol_f = model(data,data.batch.to(device))
        loss = loss_func(output, target.float())
        train_loss = loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        pred_loss += loss.item()
        all_target.append(target.detach().cpu())
        all_output.append(output.detach().cpu())
        all_mol_f.append(mol_f.detach().cpu())
    losses['pred'].append(pred_loss)
    all_target = torch.cat(all_target, dim=0)
    all_output = torch.cat(all_output, dim=0)
    #all_mol_f = torch.cat(all_mol_f, dim=0)

    return losses, all_target, all_output, all_mol_f

def evaluate(args, model, val_loader, loss_func, metric_funcs, device):
    """
    Evaluates a model on a validation set without performing backpropagation.
    """
    model.eval()
    losses = collections.defaultdict(list)
    total_loss, pred_loss = 0.0, 0.0
    y_pred, y_true = torch.zeros(0, args.num_classes), torch.zeros(0, 1)
    graph_count = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            batch = data.batch.to(device)
            target = data.target.reshape(-1, 1).to(device)
            out,mol_f = model(data, batch)
            loss = loss_func(out, target.float())
            total_loss += loss.item()*data.num_graphs
            graph_count += data.num_graphs
            y_pred = torch.cat((y_pred, out.cpu().detach().reshape(-1, args.num_classes)))
            y_true = torch.cat((y_true, target.cpu().detach()))
    
    # Calculate all metrics
    val_scores = {}
    for metric, func in metric_funcs.items():
        val_scores[metric] = func(y_true, torch.sigmoid(y_pred) if args.task == 'classification' else y_pred)

    return val_scores, total_loss/graph_count

def predict(args, model, test_loader, loss_func, device):
    """
    Evaluates a model on a test set using explanation_forward (performing backpropagation).
    """
    model.eval()
    model.to(device)
    y_pred, y_true, cliffs = torch.zeros(0, args.num_classes), torch.zeros(0, 1), torch.zeros(0, 1)
    all_mol_f = []
    for data in test_loader:
        data = data.to(device)
        batch = data.batch.to(device)
        target = data.target.reshape(-1, 1).to(device)
        output, mol_f = model(data, batch)
        loss = loss_func(output, target.float())

        y_pred = torch.cat((y_pred, output.cpu().detach().reshape(-1, args.num_classes)))
        y_true = torch.cat((y_true, target.cpu().detach()))
        #all_mol_f.append(mol_f.cpu().detach())
    #mol_f = torch.cat(all_mol_f, dim=0)
    return  y_pred, y_true, cliffs, all_mol_f


def run_cv_train(args, data_train: List[Data], dataset: MoleculeDataset, gnn_config=None):
    """
    Trains a model on a K-Fold cross-validation set, returns the best model for each fold.
    Adopt from https://github.com/shenwanxiang/ACANet/blob/main/clsar/main.py#L185 _cv_split(
    """
    from sklearn.model_selection import StratifiedKFold
    gnn_config = {
                        'num_node_features': args.num_node_features,
                        'num_edge_features': args.num_edge_features,
                        'node_hidden_dim': args.node_hidden_dim,
                        'edge_hidden_dim': args.edge_hidden_dim,
                        'num_classes': args.num_classes,
                        'conv_name': args.conv_name,
                        'num_layers': args.num_layers,
                        'hidden_dim': args.hidden_dim,
                        'dropout_rate': args.dropout_rate,
                        'pool': args.pool,
                        'heads': args.heads,
                        'uncom_pool': args.uncom_pool,
                        'embed_method': args.embed_method,
                    }       
    if args.conv_name == 'pna':
        gnn_config['deg'] = get_deg(data_train)
    KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    y_all = [data.target for data in data_train]
    cutoff = np.median(y_all)
    labels = [0 if i < cutoff else 1 for i in y_all]
    splits = [{'inner_train_idx': i, 'inner_val_idx': j} for i, j in KFold.split(labels, labels)]
    initial_fold_seed = args.seed
    for i, split in enumerate(splits):
        inner_train_data = [data_train[idx] for idx in split['inner_train_idx']]
        inner_val_data = [data_train[idx] for idx in split['inner_val_idx']]
        print(f"Training fold {i+1} with seed {args.seed}")
        if args.conv_name == 'pna':
            args.deg = get_deg(inner_train_data)
            gnn_config['deg'] = args.deg
        # Pack training data if using explanation loss (validation stays unpacked)
        if args.loss != 'MSE':
            inner_train_data = pack_data(inner_train_data, dataset.cliff_dict, pair_cap=args.pair_cap)
        args.seed = initial_fold_seed * 10 + i
        _ = run_training(args, inner_train_data, inner_val_data)
    # Restore original seed to avoid compounding on subsequent calls.
    args.seed = initial_fold_seed
    print("All folds trained")
        
        
