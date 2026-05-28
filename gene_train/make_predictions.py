from collections import OrderedDict
import csv
from typing import List, Optional, Union, Tuple

import numpy as np

from args import PredictArgs, TrainArgs

import pandas as pd
from create_data import *
from utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_reaction, set_explicit_h, set_adding_hs, \
    reset_featurization_parameters
from model import MoleculeModel
from chemprop.uncertainty import UncertaintyCalibrator, build_uncertainty_calibrator, UncertaintyEstimator, \
    build_uncertainty_evaluator


import torch
from collections import OrderedDict
from os import makedirs
from torch_geometric import data as DATA

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def load_model(args: PredictArgs, generator: bool = False):
    """
    Function to load a model or ensemble of models from file. If generator is True, a generator of the respective model and scaler
    objects is returned (memory efficient), else the full list (holding all models in memory, necessary for preloading).

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param generator: A boolean to return a generator instead of a list of models and scalers.
    :return: A tuple of updated prediction arguments, training arguments, a list or generator object of models, a list or
                 generator object of scalers, the number of tasks and their respective names.
    """
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    # Load model and scalers
    models = (
        load_checkpoint(checkpoint_path, device=args.device) for checkpoint_path in args.checkpoint_paths
    )
    scalers = (
        load_scalers(checkpoint_path) for checkpoint_path in args.checkpoint_paths
    )
    if not generator:
        models = list(models)

    return args, train_args, models, num_tasks, task_names


def load_data(args: PredictArgs, smiles: list):
    """
    Load data from CSV or SMILES list, convert to PyG Data object.
    Returns a Batch object (multiple graphs batched together).
    """
    print("Loading data")
    data_df = pd.read_csv(args.test_path)
    smiles = data_df["smiles"].tolist()

    data_list = []
    for i, smi in enumerate(smiles):
        try:
            # --- 1. sequence encoding ---
            seq = smiles_to_sequence(smi)
            # --- 2D graph ---
            atom_graph, bond_graph = create_atom_bond_graph(smi)
            atoms_feature, edge_index, edge_attr = atom_graph.get_atom_feature()
            bonds_feature, bond_index = bond_graph.get_bond_feature()
            # --- 3D graph ---
            h, x, edges_3d, edge_attr_3d = smiles_to_egnn(smi)

            # --- 检查关键张量是否为空 ---
            if len(atoms_feature) == 0 or len(edge_index) == 0 or len(bonds_feature) == 0 or len(h) == 0:
                smi = "CCO"
                seq = smiles_to_sequence(smi)
            # --- 2D graph ---
                atom_graph, bond_graph = create_atom_bond_graph(smi)
                atoms_feature, edge_index, edge_attr = atom_graph.get_atom_feature()
                bonds_feature, bond_index = bond_graph.get_bond_feature()
            # --- 3D graph ---
                h, x, edges_3d, edge_attr_3d = smiles_to_egnn(smi)

            # --- create Data object ---
            GCNData = DATA.Data(
                smiles=smi,
                atoms_features=torch.FloatTensor(atoms_feature),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_attr),
                bonds_features=torch.FloatTensor(bonds_feature),
                bond_index=torch.LongTensor(bond_index),
                seqs=torch.tensor(seq, dtype=torch.long).unsqueeze(0),
                h_3d=torch.FloatTensor(h),
                x_3d=torch.FloatTensor(x),
                edge_index_3d=torch.LongTensor(edges_3d),
                edge_attr_3d=torch.FloatTensor(edge_attr_3d)
            )

            # --- batch info ---
            GCNData.num_nodes = torch.tensor(GCNData.atoms_features.size(0), dtype=torch.long)
            num_bonds = GCNData.bonds_features.size(0)
            GCNData.bond_batch = torch.zeros(num_bonds, dtype=torch.long)
            num_3d_nodes = GCNData.h_3d.size(0)
            GCNData.batch_3d = torch.zeros(num_3d_nodes, dtype=torch.long)

            data_list.append(GCNData)

        except Exception as e:
            print(f"Error processing SMILES at index {i}: {smi}, error: {e}")
            continue

    if len(data_list) == 0:
        raise ValueError("No valid molecules found!")

    return data_list

def predict_and_save(
    args,
    train_args,
    task_names: List[str],
    num_tasks: int,
    full_data,
    models: List,
    num_models: int,
    batch_size: int = 32,
    save_results: bool = True,
):
    """
    Safe version of predict_and_save with batching.
    full_data: list[Data] or list[tuple], each element is a PyG Data object.
    """
    device = "cuda:0"
    num_samples = len(full_data)
    preds_list = []

    # 如果 full_data 是 tuple list，取第 0 个元素
    data_list = [d if isinstance(d, DATA.Data) else d[0] for d in full_data]

    # 按 batch_size 分批处理
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_list = data_list[start_idx:end_idx]

        # 生成 PyG Batch 并移动到 GPU
        batch = Batch.from_data_list(batch_list).to(device)
        batch_model_preds = []

        for model_idx, model in enumerate(models):
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                try:
                    output = model(batch)
                    if output is None:
                        print(f"Warning: model {model_idx} returned None for batch {start_idx}-{end_idx}")
                        output = torch.full(
                            (len(batch_list), num_tasks),
                            float("nan"),
                            device=device
                        )
                    else:
                        output = output.to(device)

                except Exception as e:
                    print(f"Warning: model {model_idx} failed on batch {start_idx}-{end_idx} with error: {e}")
                    output = torch.full(
                        (len(batch_list), num_tasks),
                        float("nan"),
                        device=device
                    )

            batch_model_preds.append(output)

        # ensemble 平均
        if len(batch_model_preds) == 0:
            batch_mean_pred = torch.full(
                (len(batch_list), num_tasks),
                float("nan"),
                device=device
            )
        else:
            batch_mean_pred = torch.stack(batch_model_preds, dim=0).mean(dim=0)

        preds_list.append(batch_mean_pred.cpu())

    # 拼接所有 batch 的预测
    preds = torch.cat(preds_list, dim=0).numpy()

    # 保存 CSV
    if save_results:
        import csv
        with open(args.preds_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(task_names)
            for row in preds:
                writer.writerow(row)

    return preds



def set_features(args: PredictArgs, train_args: TrainArgs):
    """
    Function to set extra options.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    """
    reset_featurization_parameters()

    if args.atom_descriptors == "feature":
        set_extra_atom_fdim(train_args.atom_features_size)

    # if args.bond_descriptors == "feature":
    #     set_extra_bond_fdim(train_args.bond_features_size)

    # set explicit H option and reaction option
    set_explicit_h(train_args.explicit_h)
    set_adding_hs(args.adding_h)
    # set_keeping_atom_map(args.keeping_atom_map)
    if train_args.reaction:
        set_reaction(train_args.reaction, train_args.reaction_mode)
    elif train_args.reaction_solvent:
        set_reaction(True, train_args.reaction_mode)






@ timeit()
def make_predictions(
        args: PredictArgs,
        smiles: List[List[str]] = None,
        model_objects: Tuple[
            PredictArgs,
            TrainArgs,
            List[MoleculeModel],
            List[Union[StandardScaler]],
            int,
            List[str],
        ] = None,
        calibrator: UncertaintyCalibrator = None,
        return_invalid_smiles: bool = True,
        return_index_dict: bool = False,
        return_uncertainty: bool = False,
) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param model_objects: Tuple of output of load_model function which can be called separately outside this function. Preloaded model objects should have
                used the non-generator option for load_model if the objects are to be used multiple times or are intended to be used for calibration as well.
    :param calibrator: A :class: `~chemprop.uncertainty.UncertaintyCalibrator` object, for use in calibrating uncertainty predictions.
                Can be preloaded and provided as a function input or constructed within the function from arguments. The models and scalers used
                to initiate the calibrator must be lists instead of generators if the same calibrator is to be used multiple times or
                if the same models and scalers objects are also part of the provided model_objects input.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :param return_index_dict: Whether to return the prediction results as a dictionary keyed from the initial data indexes.
    :param return_uncertainty: Whether to return uncertainty predictions alongside the model value predictions.
    :return: A list of lists of target predictions. If returning uncertainty, a tuple containing first prediction values then uncertainty estimates.
    """
    (args, train_args, models, num_tasks, task_names) = load_model(
            args)
    num_models = len(args.checkpoint_paths)

    set_features(args, train_args)

    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or get_invalid_smiles_from_list functions from data/utils.py
    full_data = load_data(args, smiles)

    preds = predict_and_save(
            args=args,
            train_args=train_args,
            task_names=task_names,
            num_tasks=num_tasks,
            full_data=full_data,
            models=models,
            num_models=num_models
        )

    if return_index_dict:
        preds_dict = {}
        unc_dict = {}
        for i in range(len(full_data)):
            if return_invalid_smiles:
                preds_dict[i] = preds[i]
                unc_dict[i] = unc[i]
            else:
                valid_index = full_to_valid_indices.get(i, None)
                if valid_index is not None:
                    preds_dict[i] = preds[valid_index]
                    unc_dict[i] = unc[valid_index]
    return preds

