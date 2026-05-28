import string
import pandas as pd
import numpy as np
import os
from rdkit.Chem import MolFromSmiles, AllChem
import re
from typing import List
import string
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import re

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
    
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

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
    #x = x - x.mean(dim=0, keepdim=True)
    
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