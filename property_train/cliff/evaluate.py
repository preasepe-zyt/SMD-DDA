import collections  
from typing import List
import numpy as np
from cliff.utils.metrics import get_metric_func
from cliff.dataset import MoleculeDataset
from cliff.featurization import MolTensorizer
from cliff.models.GNN import GNN
from cliff.train import predict
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import pandas as pd


fp_gen = GetMorganGenerator(radius=2, fpSize=1024)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[0] == b.shape[0] == 1
    return torch.nn.CosineSimilarity(dim=1)(a, b)
    
def get_fp(smiles):
    # 1️⃣ 检查输入
    if not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 2️⃣ 正确API
    fp = fp_gen.GetFingerprint(mol)

    # 3️⃣ 转 numpy
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr

def get_prop_dist_matrix(y: np.ndarray):
    if hasattr(y, "detach"):   # 是 torch.Tensor
        y = y.detach().cpu().numpy()
    assert y.shape[1]==1
    mat = np.abs(y - y.T)
    threshold = (1. - np.min(mat))/(np.max(mat) - np.min(mat))
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat)), threshold

def get_representation_sim_matrix(X: np.ndarray):
    # get the minmax normalized euclidean distance matrix
    dist = np.zeros((X.shape[0], X.shape[0]))
    ones = torch.ones(X.shape[0], X.shape[0])
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            dist[i, j] = np.linalg.norm(X[i] - X[j])
    return ones - (dist - np.min(dist))/(np.max(dist) - np.min(dist))


def run_evaluation(args, dataset, data_test, model, metric, xeval=False):
    """
    Runs evaluation for a single model using the specified metric.
    Returns a dictionary of all scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_func = torch.nn.MSELoss() if args.task == 'regression' else torch.nn.BCEWithLogitsLoss()
    metric_funcs = get_metric_func(metric=metric)
    scores = {}

    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    y_pred, y_true, cliffs, mol_f = predict(args, model, test_loader, loss_func, device)

    
    all_smiles = []
    y_true = torch.zeros(0, 1)
    for data in test_loader:
        smiles = data.smiles   # 一般是 list
        all_smiles.extend(smiles)
        target = data.target.reshape(-1, 1).to(device)
        y_true = torch.cat((y_true, target.cpu().detach()))
    fps = [get_fp(s) for s in all_smiles]
    dist, threshold = get_prop_dist_matrix(y_true)

    
    sim_fp = np.zeros((len(fps), len(fps)))
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
              sim_fp[i, j] = cosine_similarity(torch.Tensor(fps[i]).unsqueeze(0), torch.Tensor(fps[j]).unsqueeze(0)).item()
    #sim = np.zeros((mol_f.shape[0], mol_f.shape[0]))
#    for i in range(mol_f.shape[0]):
#        for j in range(i+1, mol_f.shape[0]):
#              sim[i, j] = cosine_similarity(torch.Tensor(mol_f[i]).unsqueeze(0), torch.Tensor(mol_f[j]).unsqueeze(0)).item()
#    df = pd.DataFrame()
#    df['sim'] = np.concatenate([sim_fp[np.triu_indices(sim_fp.shape[0], k=1)], sim[np.triu_indices(sim.shape[0], k=1)]])
#    df['label'] = np.concatenate([['ECFP']*sim_fp[np.triu_indices(sim_fp.shape[0], k=1)].shape[0], ['SMD-DDA']*sim[np.triu_indices(sim.shape[0], k=1)].shape[0]])
#    df['dist'] = np.concatenate([dist[np.triu_indices(dist.shape[0], k=1)],dist[np.triu_indices(dist.shape[0], k=1)]])
#    df['dataset'] = [args.dataset]*df.shape[0]
#   
#    
#    print("property_dist", df.shape)
#    df.to_csv("property_dist.csv", index=False)
    
    scores[f"test_{metric}"] = metric_funcs(y_true, torch.sigmoid(y_pred) if args.task == 'classification' else y_pred)
    print('test {:.4s}: {:.3f}'.format(metric, scores[f"test_{metric}"]))    
    if cliffs.sum() > 0:
        y_pred_cliff = y_pred[cliffs==1]
        y_true_cliff = y_true[cliffs==1]
        if sum(y_true_cliff) == 0 or sum(y_true_cliff) == len(y_true_cliff):
            for metric in args.metric:
                scores[f"test_cliff_{metric}"] = 0
        else:
            scores[f"test_cliff_{metric}"] = metric_funcs(y_true_cliff, y_pred_cliff)
            print('test cliff {:.4s}: {:.3f}'.format(metric, scores[f"test_cliff_{metric}"]))
    print("scores: ", scores)
    return scores

def run_evaluation_ensemble(args, dataset: MoleculeDataset, data_test: Data, models, metric, xeval=False):
    """
    Runs evaluation for an ensemble of models.
    Returns a dictionary of all scores.
    """
    print(f"Starting run_evaluation_ensemble with xeval={xeval}, {len(models)} models", flush=True)
    for model in models:
        model.eval()
    metric_funcs = get_metric_func(metric=metric)
    all_scores = collections.defaultdict(list)
    print(f"Packing test data (this may take a while for large datasets)...", flush=True)
    y_preds, y_trues = [], []
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = torch.nn.MSELoss() if args.task == 'regression' else torch.nn.BCEWithLogitsLoss()
    for j in range(5):
        print(f"Running prediction for model {j+1}/5...", flush=True)
        model = models[j]
        model.to(device)
        y_pred, y_true, cliffs = predict(args, model, test_loader, loss_func, device)
        y_preds.append(y_pred)
        y_trues.append(y_true)
    print(f"All predictions completed, computing ensemble metrics...", flush=True)
    y_preds = torch.cat(y_preds, dim=1)
    y_trues = torch.cat(y_trues, dim=1)
    y_ensemble_pred = torch.mean(y_preds, dim=1)
    # Ground truth is the same for all models, use first one (don't average)
    y_ensemble_true = y_trues[:, 0]
    cliffs = np.array([cliffs[i].item() for i in range(len(cliffs))])
    y_ensemble_pred_cliff = y_ensemble_pred[cliffs==1]
    y_ensemble_true_cliff = y_ensemble_true[cliffs==1]
    all_scores[f"test_{metric}"] = metric_funcs(y_ensemble_true, y_ensemble_pred)
    print('test {:.4s}: {:.3f}'.format(metric, all_scores[f"test_{metric}"]), flush=True)
    all_scores[f"test_cliff_{metric}"] = metric_funcs(y_ensemble_true_cliff, y_ensemble_pred_cliff)
    print('test cliff {:.4s}: {:.3f}'.format(metric, all_scores[f"test_cliff_{metric}"]), flush=True)
    print(f"run_evaluation_ensemble completed!", flush=True)
    return all_scores
        
