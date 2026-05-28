import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os
import random
from typing import List, Union, Optional, Tuple
from copy import deepcopy
from torch_geometric.data import Batch, Data
from cliff.featurization import MolTensorizer
from cliff.cliffs import ActivityCliffs, get_tanimoto_matrix
from cliff.utils.const import DATASETS, MOLDATASETS
import torch
import deepchem as dc

class MoleculeDataset:
    def __init__(
        self, 
        file: str, 
        working_dir: str = None):
        """ 
        Data class to easily load featurized molecular data, including activity cliff information
        """

        if os.path.exists(file):
            # If a direct CSV path is provided, remember its location for outputs (e.g., mcs_dict).
            self.dataset_name = os.path.splitext(os.path.basename(file))[0]
            self.working_path = os.path.dirname(os.path.abspath(file)) or "."
            df = pd.read_csv(file)     
        else:
            self.dataset_name = file
            if self.dataset_name in DATASETS or self.dataset_name in MOLDATASETS:
                assert working_dir is not None, "Please specify a working directory"
                self.working_path = os.path.join(working_dir, self.dataset_name)
                file = os.path.join(self.working_path, f"{self.dataset_name}.csv")
                print(f"Loading dataset {self.dataset_name} from {file}")
                df = pd.read_csv(file)
            else:
                print(f"Dataset {self.dataset_name} not found in {working_dir}")
        self.smiles_all = df['smiles'].tolist()
        self.y_all = df['y'].tolist()
        self.cliff_mols = None

        self.featurize_data()
        
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
                                    self.y_all,
                                    struct_sim=struct_sim, 
                                    dist_thre=dist_thre, 
                                    dict_path=dict_path)
        self.cliff_mols = self.cliff.cliff_mols     
        print(f"Found {sum(self.cliff_mols)} cliffs in dataset {self.dataset_name}")
        self.cliff_dict = self.cliff.mcs_dict
        for i in range(len(self.data_all)):
            self.data_all[i].cliff = self.cliff_mols[i]

    def split_data(self,
                split_ratio: List[float] = [0.8, 0.1, 0.1],
                split_method: str = 'random',
                n_clusters: Optional[int] = 5,
                seed: int = 42,
                save_split: bool = False,
                return_idx: bool = False):

        ratio = "".join([str(int(r*10)) for r in split_ratio])
        split_path = os.path.join(self.working_path, f"{self.dataset_name}_{split_method}_{ratio}_{seed}.csv")
        exists = os.path.exists(split_path)
        if exists:
            print(f"Loading split from {split_path}")
            df = pd.read_csv(split_path)
            train_idx, val_idx, test_idx = df[df['split'] == 'train'].index.tolist(), df[df['split'] == 'val'].index.tolist(), df[df['split'] == 'test'].index.tolist()
        elif split_method == 'random':
            train_idx, test_idx = train_test_split(range(len(self.smiles_all)), test_size=split_ratio[2], random_state=seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=seed)
        elif split_method == 'cliff':
            assert self.cliff_mols is not None, "No cliff information available"
            train_idx, val_idx, test_idx = cliff_split(self.smiles_all, self.y_all, self.cliff_mols, split_ratio=split_ratio, n_clusters=n_clusters, seed=seed)
        elif split_method == 'scaffold':
            pseudo_dataset = dc.data.DiskDataset.from_numpy(X=np.zeros((len(self.smiles_all))), y=np.zeros(len(self.smiles_all)), ids=self.smiles_all)
            scaffoldsplitter = dc.splits.ScaffoldSplitter()
            train_idx, val_idx, test_idx = scaffoldsplitter.split(pseudo_dataset, seed=seed, frac_train=split_ratio[0], frac_valid=split_ratio[1], frac_test=split_ratio[2])
        else:
            raise ValueError(f"Split method {split_method} not recognized")  

        if not exists and save_split:        
            split = []
            for i in range(len(self.smiles_all)):
                if i in train_idx:
                    split.append('train')
                elif i in val_idx:
                    split.append('val')
                elif i in test_idx:
                    split.append('test')
                else:
                    raise ValueError(f"Can't find molecule {i} in train, val or test")
            df = pd.DataFrame({'smiles': self.smiles_all,
                            'y': self.y_all,
                            'cliff_mol': self.cliff_mols,
                            'split': split})
            df.to_csv(split_path, index=False)
            print(f"Saved split to {split_path}")
    
        if return_idx == True:
            return train_idx, val_idx, test_idx
        else:
            data_train, data_val, data_test = [self.data_all[i] for i in train_idx], [self.data_all[i] for i in val_idx], [self.data_all[i] for i in test_idx]
            return data_train, data_val, data_test

    def featurize_data(self):
        featurizer = MolTensorizer()
        self.data_all = [featurizer.tensorize(smi) for smi in tqdm(self.smiles_all)]
        # concatenate data with smiles, target and whether or not cliff_mol.
        for i in range(len(self.data_all)):
            self.data_all[i].smiles = self.smiles_all[i]
            self.data_all[i].target = self.y_all[i]
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

