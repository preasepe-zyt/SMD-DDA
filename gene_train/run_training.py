import json
import logging as logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn

from evaluate import evaluate, evaluate_predictions
from train import train
from loss_functions import get_loss_func
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from model import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean
from torch_geometric.loader import DataLoader
from evaluate import predict

def run_training(args: TrainArgs,train_data,test_data):
    torch.manual_seed(args.pytorch_seed)



    args.train_data_size = len(train_data)

    logger.info(f'train size = {len(train_data):,} |  test size = {len(test_data):,}')

    if len(test_data) == 0:
        logger.info('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')
        empty_test_set = True
    else:
        empty_test_set = False

    # Get loss function
    loss_func = get_loss_func(args)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.class_balance:
        logger.info(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            logger.info(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            logger.info(f'Building model {model_idx}')
            model = MoleculeModel(args)

        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            logger.info(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

        logger.info(model)

        if args.checkpoint_frzn is not None:
            logger.info(f'Number of unfrozen parameters = {param_count(model):,}')
            logger.info(f'Total number of parameters = {param_count_all(model):,}')
        else:
            logger.info(f'Number of parameters = {param_count_all(model):,}')

        if args.cuda:
            logger.info('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            logger.info(f'Epoch {epoch}')
            n_iter = train(
                model=model,
                data_loader=train_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=test_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type
            )

#            for metric, scores in val_scores.items():
#                # Average validation score\
#                mean_val_score = multitask_mean(scores, metric=metric)
#                logger.info(f'Validation {metric} = {mean_val_score:.6f}')
#                writer.add_scalar(f'validation_{metric}', mean_val_score, n_iter)
#
#                if args.show_individual_scores:
#                    # Individual validation scores
#                    for task_name, val_score in zip(args.task_names, scores):
#                        logger.info(f'Validation {task_name} {metric} = {val_score:.6f}')
#                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            mean_val_score = multitask_mean(val_scores[args.metric], metric=args.metric)
            if args.minimize_score and mean_val_score < best_score or \
                    not args.minimize_score and mean_val_score > best_score:
                best_score, best_epoch = mean_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model,args)

        # Evaluate on test set using model with best validation score
        logger.info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        if empty_test_set:
            logger.info(f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
        else:
            test_preds, test_targets = predict(
                model=model,
                data_loader=test_loader
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                logger=logger
            )

            if len(test_preds) != 0:
                if 'sum_test_preds' not in locals():
                    sum_test_preds = np.array(test_preds)
                else:
                    sum_test_preds += np.array(test_preds)

            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                logger.info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                writer.add_scalar(f'test_{metric}', avg_test_score, 0)

                if args.show_individual_scores and args.dataset_type != 'spectra':
                    # Individual test scores
                    for task_name, test_score in zip(args.task_names, scores):
                        logger.info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                        writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

    # Evaluate ensemble on test set
    if empty_test_set:
        ensemble_scores = {
            metric: [np.nan for task in args.task_names] for metric in args.metrics
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            logger=logger
        )
#
#    for metric, scores in ensemble_scores.items():
#        # Average ensemble score
#        mean_ensemble_test_score = multitask_mean(scores, metric=metric)
#        logger.info(f'Ensemble test {metric} = {mean_ensemble_test_score:.6f}')
#
#        # Individual ensemble scores
#        if args.show_individual_scores:
#            for task_name, ensemble_score in zip(args.task_names, scores):
#                logger.info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Save scores
    # 计算 summary    ensemble_scores_summary = {}
    ensemble_scores_summary = {}
    for k, v in ensemble_scores.items():
        v = np.array(v)

        ensemble_scores_summary[f"{k}_mean"] = float(np.mean(v))
        ensemble_scores_summary[f"{k}_std"]  = float(np.std(v))
        ensemble_scores_summary[k] = v.tolist() 
    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores_summary, f, indent=4, sort_keys=True)


    # Optionally save test preds
    if args.save_preds and not empty_test_set:
        all_smiles = [d.smiles for d in test_data]
        test_preds_dataframe = pd.DataFrame(data={'smiles': all_smiles})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores

