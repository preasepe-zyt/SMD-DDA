import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, MolFromSmiles, AllChem
from rdkit.Chem.rdchem import Atom, Bond, Mol
from torch_geometric.data import Data
from typing import List, Union
import numpy as np
from cliff.utils.const import ATOM_TYPES, BOND_TYPES, STEREO_TYPES

import pandas as pd
import os
import json,pickle
from collections import OrderedDict
import networkx as nx
import re
from typing import List
import string
from torch.nn.utils.rnn import pad_sequence
import re


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def smiles_to_sequence(smiles, max_len=138, charset=None):
    if charset is None:
        charset = list(string.ascii_letters + string.digits + "()-=#$@+/\\")  # 你可根据需要扩展字符集
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}  # 0留给padding
    sequence = [char_to_idx.get(char, 0) for char in smiles]
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return np.array(sequence, dtype=np.int32)

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]
    
class AtomGraph:
    def __init__(self, mol):
        self.mol = mol
        self.build()

    def atom_feature(self, atom):
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + #Atom symbol
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Number of adjacent atoms
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # Number of adjacent hydrogens
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Implicit valence
                        one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + #Formal charge
                        one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) + #Hybridization
                        [atom.GetIsAromatic()] + #Aromaticity
                        [atom.IsInRing()] #In ring
                        )

    def bond_feature(self, bond):
        bt = bond.GetBondType()

        if bt == Chem.rdchem.BondType.SINGLE:
            return [1, 0, 0, 0]
        elif bt == Chem.rdchem.BondType.DOUBLE:
            return [0, 1, 0, 0]
        elif bt == Chem.rdchem.BondType.TRIPLE:
            return [0, 0, 1, 0]
        elif bond.GetIsAromatic():
            return [0, 0, 0, 1]
        else:
            return [0, 0, 0, 0]

    def build(self):
        mol = self.mol

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_feature(atom))

        self.atom_features = np.array(atom_features, dtype=np.float32)

        edge_list = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feat = self.bond_feature(bond)

            # 双向边
            edge_list.append([i, j])
            edge_list.append([j, i])

            edge_attr.append(feat)
            edge_attr.append(feat)

        if len(edge_list) == 0:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
            self.edge_attr = np.zeros((0, 4), dtype=np.float32)
        else:
            self.edge_index = np.array(edge_list, dtype=np.int64).T
            self.edge_attr = np.array(edge_attr, dtype=np.float32)

    def get_atom_feature(self):
        return self.atom_features, self.edge_index, self.edge_attr


class BondGraph:
    """
    每条有向键作为一个节点
    """

    def __init__(self, mol):
        self.mol = mol
        self.build()

    def bond_feature(self, bond):
        bt = bond.GetBondType()

        if bt == Chem.rdchem.BondType.SINGLE:
            return [1, 0, 0, 0]
        elif bt == Chem.rdchem.BondType.DOUBLE:
            return [0, 1, 0, 0]
        elif bt == Chem.rdchem.BondType.TRIPLE:
            return [0, 0, 1, 0]
        elif bond.GetIsAromatic():
            return [0, 0, 0, 1]
        else:
            return [0, 0, 0, 0]

    def build(self):
        mol = self.mol

        bond_features = []
        bond_index = []
        edge_map = {}

        idx = 0

        # 构造有向键节点
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # 正向
            edge_map[(i, j)] = idx
            bond_features.append(self.bond_feature(bond))
            idx += 1

            # 反向
            edge_map[(j, i)] = idx
            bond_features.append(self.bond_feature(bond))
            idx += 1

        # 构造 bond graph 的边（共享中间原子）
        for (i, j), idx1 in edge_map.items():
            for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
                k = neighbor.GetIdx()
                if k == i:
                    continue
                idx2 = edge_map.get((j, k))
                if idx2 is not None:
                    bond_index.append([idx1, idx2])

        if len(bond_features) == 0:
            self.bond_features = np.zeros((0, 4))
            self.bond_index = np.zeros((2, 0), dtype=np.int64)
        else:
            self.bond_features = np.array(bond_features, dtype=np.float32)
            if len(bond_index) == 0:
                self.bond_index = np.zeros((2, 0), dtype=np.int64)
            else:
                self.bond_index = np.array(bond_index, dtype=np.int64).T

    def get_bond_feature(self):
        return self.bond_features, self.bond_index


def create_atom_bond_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_graph = AtomGraph(mol)
    bond_graph = BondGraph(mol)
    return atom_graph, bond_graph


def smiles_to_egnn(smiles):
    """
    输入: 单个 SMILES
    输出: h, x, edges, edge_attr (torch tensors)
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # 节点特征 h
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    max_atomic_num = 100
    h_np = np.zeros((len(atom_types), max_atomic_num))
    for i, z in enumerate(atom_types):
        h_np[i, z-1] = 1
    h = torch.tensor(h_np, dtype=torch.float32)
    
    # 节点坐标 x
    try:
        conf = mol.GetConformer()
        coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )
    except Exception as e:
    # 用 1 填充 (N_atoms, 3)
        coords = np.ones((mol.GetNumAtoms(), 3))
    x = torch.tensor(coords, dtype=torch.float32)
    # 归一化坐标

    
    # 边和边特征
    edges_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges_list.append([i, j])
        edges_list.append([j, i])
        
        # bond_type one-hot: [single, double, triple, aromatic]
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            feat = [1, 0, 0, 0]
        elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            feat = [0, 1, 0, 0]
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            feat = [0, 0, 1, 0]
        elif bond.GetIsAromatic():
            feat = [0, 0, 0, 1]
        else:
            feat = [0, 0, 0, 0]  # 其他特殊键

        edge_attr_list.append(feat)
        edge_attr_list.append(feat)  # 双向

    edges = torch.tensor(edges_list, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    
    return h, x, edges, edge_attr




def one_hot_encoding(x, allowable_set):
    """One-hot encoding.
    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    Returns
    -------
    list
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1, then we must have x == allowable_set[i].
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(int, list(map(lambda s: x == s, allowable_set))))


def multi_hot_encoding(x, allowable_set):
    """Multi-hot encoding.
    Args:
        x (list): any type that can be compared with elements in allowable_set
        allowable_set (list): allowed values for x to take
    Returns:
        list: List of int (0 or 1) where zero or more values can be 1.
            If the i-th value is 1, then we must have allowable_set[i] in x.
    """
    return list(map(int, list(map(lambda s: s in x, allowable_set))))

def get_pos(mol: Mol) -> torch.Tensor:
    AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    N = mol.GetNumAtoms()
    pos = Chem.MolToMolBlock(mol).split('\n')[4:4 + N]
    pos = [[float(x) for x in line.split()[:3]] for line in pos]
    return torch.tensor(pos)

class MolTensorizer(object):
    def __init__(self, 
                 featurization='normal',
                 ):
        self.featurization = featurization
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        self.num_node_feature = len(self.atom_features(unrelated_mol.GetAtomWithIdx(0)))
        self.num_bond_feature = len(self.bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        
        self.smiles_to_sequence = smiles_to_sequence
        self.create_atom_bond_graph = create_atom_bond_graph
        self.smiles_to_egnn = smiles_to_egnn
        

    def atom_features(
        self, atom: Atom, use_chirality: bool = False, hydrogens_implicit: bool = False
    ) -> List[float]:
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms
        atom_types = ATOM_TYPES
        if hydrogens_implicit == True:
            atom_types = ["H"] + atom_types
        # compute atom features
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), atom_types)
        implicit_valence_enc = one_hot_encoding(
            int(atom.GetImplicitValence()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        n_heavy_neighbors_enc = one_hot_encoding(
            int(atom.GetDegree()), [1, 2, 3, 4, "MoreThanFour"]
        )
        formal_charge_enc = one_hot_encoding(
            int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
        )
        hybridisation_type_enc = one_hot_encoding(
            str(atom.GetHybridization()),
            ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
        )

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

        atom_feature_vector = (
            atom_type_enc
            + implicit_valence_enc
            + n_heavy_neighbors_enc
            + formal_charge_enc
            + hybridisation_type_enc
            + is_in_a_ring_enc
            + is_aromatic_enc
            + atomic_mass_scaled
            + vdw_radius_scaled
            + covalent_radius_scaled
        )

        if use_chirality:
            chirality_type_enc = one_hot_encoding(
                str(atom.GetChiralTag()),
                [
                    "CHI_UNSPECIFIED",
                    "CHI_TETRAHEDRAL_CW",
                    "CHI_TETRAHEDRAL_CCW",
                    "CHI_OTHER",
                ],
            )
            atom_feature_vector += chirality_type_enc
        if hydrogens_implicit == True:
            n_hydrogens_enc = one_hot_encoding(
                int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
            )
            atom_feature_vector += n_hydrogens_enc
        return np.array(atom_feature_vector)
        
    def bond_features(self, bond: Bond, use_stereochemistry: bool = False) -> np.ndarray:
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """

        bond_type_enc = one_hot_encoding(str(bond.GetBondType()), BOND_TYPES)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [float(int(bond.IsInRing()))]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = one_hot_encoding(
                str(bond.GetStereo()), STEREO_TYPES
            )
            bond_feature_vector += stereo_type_enc

        return np.array(bond_feature_vector)
    
    def tensorize(self, smile: Union[str, Mol]) -> Data:
        def safe_mol(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"Could not parse SMILES: {smi}")
                return mol
            except:
            # 如果解析失败，用最简单的 methane 代替
                return Chem.MolFromSmiles("C")  

    # Step1: 解析分子
        if isinstance(smile, str):
            mol = safe_mol(smile)
        else:
            mol = smile

    # Step2: 序列
        seq = self.smiles_to_sequence(smile)

    # Step3: 构建 atom/bond 图
        atom_g, bond_g = self.create_atom_bond_graph(smile)
        atoms_feature, edge_index, edge_attr = atom_g.get_atom_feature()
        bonds_feature, bond_index = bond_g.get_bond_feature()

    # Step4: 3D 图
        three_d_g = self.smiles_to_egnn(smile)
        h, x, edges_3d, edge_attr_3d = three_d_g

    # Step5: 检查空图，如果没有节点或边就用最简单 methane
        if len(atoms_feature) == 0 or len(bonds_feature) == 0 or len(h) == 0:
            mol = Chem.MolFromSmiles("CCO")  # methane
            seq = self.smiles_to_sequence("CCO")
            atom_g, bond_g = self.create_atom_bond_graph("CCO")
            atoms_feature, edge_index, edge_attr = atom_g.get_atom_feature()
            bonds_feature, bond_index = bond_g.get_bond_feature()
            h, x, edges_3d, edge_attr_3d = self.smiles_to_egnn("CCO")

    # Step6: 构造 batch index
        num_nodes = len(atoms_feature)
        num_bonds = len(bonds_feature)
        num_3d_nodes = len(h)
        bond_batch = torch.zeros(num_bonds, dtype=torch.long)
        batch_3d = torch.zeros(num_3d_nodes, dtype=torch.long)

    # Step7: 构造 Data 对象
        data = Data(
        smiles=smile,
        atoms_features=torch.FloatTensor(atoms_feature),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        bonds_features=torch.FloatTensor(bonds_feature),
        bond_index=torch.LongTensor(bond_index),
        seqs=torch.tensor(seq, dtype=torch.long).unsqueeze(0),
        h_3d=torch.FloatTensor(h),
        x_3d=torch.FloatTensor(x),
        edge_index_3d=torch.LongTensor(edges_3d),
        edge_attr_3d=torch.FloatTensor(edge_attr_3d),
        bond_batch=bond_batch,
        batch_3d=batch_3d
    )
        data.num_nodes = num_nodes
        return data

 