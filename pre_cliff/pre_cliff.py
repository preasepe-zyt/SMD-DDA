import collections  
from typing import List
import numpy as np
from .featurization import MolTensorizer
from .GNN import GNN
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from .parsing import get_args
from typing import List, Union, Optional, Tuple
import pandas as pd
import os
#from cliffs import ActivityCliffs, get_tanimoto_matrix


class MoleculeDataset:
    def __init__(
        self, name):



        df = pd.read_csv(name)     

        self.smiles_all = df['smiles'].tolist()
        self.cliff_mols = None

        self.featurize_data()
        self.working_path = "./"
    def get_cliffs(self,
                   struct_sim: Union[Tuple[str, float], str, None] = ('combined', 0.9),
                   dist_thre: float = 1.0,
                   dict_path: Optional[str] = None):
        """
        Load an existing cliff dictionary if provided/available; otherwise generate.
        - If dict_path is given, use it directly.
        - Else, build the default path from struct_sim/dist_thre (same as before).
        """
        descriptor = struct_sim[0] if isinstance(struct_sim, tuple) else struct_sim
        sim_thre = struct_sim[1] if isinstance(struct_sim, tuple) else None
        if dict_path is None:
            if descriptor == 'default_mcs':
                dict_path = os.path.join(self.working_path, f'mcs_dict_{sim_thre}_default.pkl' if dist_thre==1.0 else f'mcs_dict_{sim_thre}_{dist_thre}_default.pkl')
            elif descriptor == 'mmp':
                dict_path = os.path.join(self.working_path, f'mcs_dict_mmp.pkl' if dist_thre==1.0 else f'mcs_dict_{dist_thre}_mmp.pkl')
            else:
                dict_path = os.path.join(self.working_path, f'mcs_dict_{sim_thre}.pkl' if dist_thre==1.0 else f'mcs_dict_{sim_thre}_{dist_thre}.pkl')
        self.cliff = ActivityCliffs(self.smiles_all, 
                                    struct_sim=struct_sim, 
                                    dist_thre=dist_thre, 
                                    dict_path=dict_path)
        self.cliff_mols = self.cliff.cliff_mols     
        print(f"Found {sum(self.cliff_mols)} cliffs in dataset {self.dataset_name}")
        self.cliff_dict = self.cliff.mcs_dict
        for i in range(len(self.data_all)):
            self.data_all[i].cliff = self.cliff_mols[i]

    def featurize_data(self):
        featurizer = MolTensorizer()
        self.data_all = [featurizer.tensorize(smi) for smi in tqdm(self.smiles_all)]
        # concatenate data with smiles, target and whether or not cliff_mol.
        for i in range(len(self.data_all)):
            self.data_all[i].smiles = self.smiles_all[i]
            if self.cliff_mols is not None:
                self.data_all[i].cliff = self.cliff_mols[i]

def cliff_split(smiles_all,
                y_all,
                cliff_mols,
                split_ratio: List[float] = [0.8, 0.1, 0.1],
                n_clusters: int = 5, 
                seed: int = 42):
        """
        Split data into train/val/test according to activity cliffs. Adpated from "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs"
        """
        from sklearn.cluster import SpectralClustering

        # Perform spectral clustering on a tanimoto distance matrix
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='precomputed')
        clusters = spectral.fit(get_tanimoto_matrix(smiles_all)).labels_
        train_idx, val_idx, test_idx = [], [], []
        for cluster in range(n_clusters):
                cluster_idx = np.where(clusters == cluster)[0]
                clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]
                # Can only split stratiefied on cliffs if there are at least 3 cliffs present, else do it randomly
                if sum(clust_cliff_mols) > 3:
                    clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=split_ratio[2],
                                                                    random_state=seed,
                                                                    stratify=clust_cliff_mols, shuffle=True)
                    clust_train_idx, clust_val_idx = train_test_split(clust_train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
                                                                    random_state=seed,
                                                                    stratify=[cliff_mols[i] for i in clust_train_idx], shuffle=True)
                else:
                    clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=split_ratio[2],
                                                                    random_state=seed,
                                                                    shuffle=True)
                    clust_train_idx, clust_val_idx = train_test_split(clust_train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
                                                                    random_state=seed,
                                                                    shuffle=True)
    
                train_idx.extend(clust_train_idx)
                val_idx.extend(clust_val_idx)
                test_idx.extend(clust_test_idx)

        return train_idx, val_idx, test_idx


def predict(weight_p,args, data, device):
    """
    Evaluates a model on a test set using explanation_forward (performing backpropagation).
    """
    checkpoint = torch.load(weight_p, map_location=device)
    model = GNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    y_pred = torch.zeros(0, args.num_classes)
    output,_ = model(data)
    y_pred = torch.cat((y_pred, output.cpu().detach().reshape(-1, args.num_classes)))
    return  y_pred

if __name__ == "__main__":
    dat = MoleculeDataset("./Cdataset_drugs.csv")
    sim_struct = 'combined'
    loader = DataLoader(
    dat.data_all,
    batch_size=32,
    shuffle=False
)
    args = get_args()
    device = "cuda:1"
    weight_list = os.listdir("./weight")
    
    pre_all = []
    for weight_p in weight_list:   # 外层循环（多个模型 / 多个权重）
        pre_list = []
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                y_pred = predict(os.path.join("./weight", weight_p), args, batch, device)
                pre_list.append(y_pred)
        pre_list = torch.cat(pre_list, dim=0)   # [N, 1]
        pre_all.append(pre_list)
    pre_all = torch.cat(pre_all, dim=1)   # [N, num_models]
    print(pre_all.shape)