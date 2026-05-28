from collections import defaultdict
import logging
from typing import Dict, List
import numpy as np
import torch
from chemprop.data import MoleculeDataLoader, StandardScaler
from model import MoleculeModel
from chemprop.train import get_metric_func
from scipy.stats import pearsonr

def predict(model, data_loader, device=None):

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:

            batch = batch.to(device)

            preds = model(batch)

            # ⭐ 收集预测
            all_preds.append(preds.cpu())

            # ⭐ 收集真实值
            targets = torch.tensor(
                batch.genes,
                dtype=torch.float32
            )

            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_preds, all_targets
    

def evaluate_predictions(preds,
                         targets,
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         gt_targets: List[List[bool]] = None,
                         lt_targets: List[List[bool]] = None,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param gt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param lt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    info = logger.info if logger is not None else print
    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    # Filter out empty targets for most data types, excluding dataset_type spectra
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    if dataset_type != 'spectra':
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(len(preds)):
                if targets[j][i] is not None:  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])

    # Compute metric. Spectra loss calculated for all tasks together, others calculated for tasks individually.
    results = defaultdict(list)
    if dataset_type == 'spectra':
        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(preds, targets))
    else:
        for i in range(num_tasks):
            # # Skip if all targets or preds are identical, otherwise we'll crash during classification
            if dataset_type == 'classification':
                nan = False
                if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                    nan = True
                    info('Warning: Found a task with targets all 0s or all 1s')
                if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                    nan = True
                    info('Warning: Found a task with predictions all 0s or all 1s')

                if nan:
                    for metric in metrics:
                        results[metric].append(float('nan'))
                    continue

            if len(valid_targets[i]) == 0:
                continue

            for metric, metric_func in metric_to_func.items():
                if dataset_type == 'multiclass' and metric == 'cross_entropy':
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                    labels=list(range(len(valid_preds[i][0])))))
                elif metric in ['bounded_rmse', 'bounded_mse', 'bounded_mae']:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], gt_targets[i], lt_targets[i]))
                else:
                    target = valid_targets[i]
                    pred   = valid_preds[i]
                    target = np.array([t.detach().cpu().item() if isinstance(t, torch.Tensor) else t for t in target])
                    pred = np.array([t.detach().cpu().item() if isinstance(t, torch.Tensor) else t for t in pred])
                    results[metric].append(metric_func(target, pred))

    return results



def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    # Inequality targets only need for evaluation of certain regression metrics
    if any(m in metrics for m in ['bounded_rmse', 'bounded_mse', 'bounded_mae']):
        gt_targets = data_loader.gt_targets
        lt_targets = data_loader.lt_targets
    else:
        gt_targets = None
        lt_targets = None


        

    preds, targets = predict(
        model,
        data_loader
    )


    results = evaluate_predictions(
        preds=preds.tolist(),
        targets=targets.tolist(),
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        logger=logger,
        gt_targets=gt_targets,
        lt_targets=lt_targets,
    )
    rmse_list = results['rmse']
    mean_rmse = np.mean(rmse_list)
    print(f'RMSE_test = {mean_rmse}')
    return results