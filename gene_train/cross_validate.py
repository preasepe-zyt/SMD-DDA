from collections import defaultdict
import csv
import json
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple
import subprocess

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
from utils import create_logger, makedirs, timeit, multitask_mean
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_reaction, \
    reset_featurization_parameters

import numpy as np
import pandas as pd

from run_training import run_training
from utils import create_logger, makedirs, timeit, multitask_mean
import pickle

from create_data import  *
import sys



@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param train_func: Function which runs training.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    try:
        args.save(os.path.join(args.save_dir, 'args.json'))
    except subprocess.CalledProcessError:
        debug('Could not write the reproducibility section of the arguments to file, thus omitting this section.')
        args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    # set explicit H option and reaction option
    reset_featurization_parameters(logger=logger)
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    if args.reaction:
        set_reaction(args.reaction, args.reaction_mode)
    elif args.reaction_solvent:
        set_reaction(True, args.reaction_mode)

    # Get data
    debug('Loading data')
    name = args.data_path.replace(".csv", "")
    data = TestbedDataset(root="data", dataset=f"{name}")
    debug(f'Number of tasks = {args.num_tasks}')

    if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
        raise ValueError(
            'The number of provided target weights must match the number and order of the prediction tasks')

    # Run training on different random seeds for each fold
    all_scores = defaultdict(list)
    from sklearn.model_selection import train_test_split
    for i in range(args.num_folds):
        train_data, test_data = train_test_split(data, test_size=0.20)
        
        info(f'Fold {i}')
        args.seed = init_seed + i
        args.save_dir = os.path.join(save_dir, f'fold_{i}')
        makedirs(args.save_dir)
   
        model_scores = run_training(args, train_data, test_data)
        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
    all_scores = dict(all_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    contains_nan_scores = False
    for fold_num in range(args.num_folds):
        for metric, scores in all_scores.items():
            fold_scores = scores[fold_num]
            info(f'Seed {init_seed + fold_num} ==> test {metric} = '
            f'{np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(args.task_names, scores[fold_num]):
                    #info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {metric} = {score:.6f}')
                    if np.isnan(score):
                        contains_nan_scores = True

    # Report scores across folds
    for metric, scores in all_scores.items():
        #avg_scores = multitask_mean(scores, axis=1, metric=metric)  # average score for each model across tasks
        mean_score, std_score = np.mean(scores), np.std(scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.mean(scores[:, task_num]):.6f} +/- {np.std(scores[:, task_num]):.6f}')

    if contains_nan_scores:
        info("The metric scores observed for some fold test splits contain 'nan' values. \
            This can occur when the test set does not meet the requirements \
            for a particular metric, such as having no valid instances of one \
            task in the test set or not having positive examples for some classification metrics. \
            Before v1.5.1, the default behavior was to ignore nan values in individual folds or tasks \
            and still return an overall average for the remaining folds or tasks. The behavior now \
            is to include them in the average, converting overall average metrics to 'nan' as well.")

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)

        if args.dataset_type == 'spectra':  # spectra data type has only one score to report
            row = ['spectra']
            for metric, scores in all_scores.items():
                task_scores = scores[:, 0]
                mean, std = np.mean(task_scores), np.std(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
        else:  # all other data types, separate scores by task
            for task_num, task_name in enumerate(args.task_names):
                row = [task_name]
                for metric, scores in all_scores.items():
                    task_scores = scores[:, task_num]
                    mean, std = np.mean(task_scores), np.std(task_scores)
                    row += [mean, std] + task_scores.tolist()
                writer.writerow(row)

    # Determine mean and std score of main metric
    avg_scores = multitask_mean(all_scores[args.metric], metric=args.metric, axis=1)
    mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)

    # Optionally merge and save test preds
    if args.save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                               for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score

