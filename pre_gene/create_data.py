import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem
import networkx as nx
from .utils import *
import re
from typing import List
import string
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import data as DATA
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
    
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='', smi=None, gene=None,
                 atom_graph=None, bond_graph=None, smile_sequences=None,
                 three_d_graph=None):
        # root 用于保存预处理数据，默认 '/tmp'
        super(TestbedDataset, self).__init__(root)
        self.dataset = dataset.replace(".csv", "")
        self.pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')

        if os.path.isfile(self.processed_paths[0]):
            print(f'Pre-processed data found: {self.processed_paths[0]}, loading ...')
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(f'Pre-processed data {self.processed_paths[0]} not found, doing pre-processing...')
            self.process_data(smi, gene, atom_graph, bond_graph, smile_sequences, three_d_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 如果没有原始文件，可返回空列表
        return []

    @property
    def processed_file_names(self):
        return [f"{self.dataset or 'data'}.pt"]

    def download(self):
        # 下载到 self.raw_dir，如果需要的话
        pass
    def refresh_data(self):
        for data in self:
            for attr in ['genes', 'atoms_features', 'bonds_features', 'seqs',
                     'h_3d', 'x_3d', 'edge_attr_3d', 'edge_index',
                     'bond_index', 'edge_attr', 'edge_index_3d', 'y']:
                original_attr = f'_original_{attr}'
                if hasattr(data, original_attr):
                    setattr(data, attr, getattr(data, original_attr).clone())
    def process_data(self, smi, gene, atom_graph, bond_graph, smile_sequences, three_d_graph):
        # 确保所有输入列表长度一致
        assert (len(smi) == len(gene) and
                len(atom_graph) == len(bond_graph) == len(smile_sequences) == len(three_d_graph)), \
            "All input lists must have the same length!"

        data_list = []
        data_len = len(smi)

        # 创建 processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        for i in range(data_len):
            try:
                print(f'Preparing data in PyTorch Format: {i+1}/{data_len}')
                smiles = smi[i]
                genes = gene[i]

                # 获取原子和键特征
                atoms_feature, edge_index, edge_attr = atom_graph[smiles].get_atom_feature()
                bonds_feature, bond_index = bond_graph[smiles].get_bond_feature()
                seq = smile_sequences[smiles]

                # 获取 3D 图特征
                h, x, edges_3d, edge_attr_3d = three_d_graph[smiles]

                # 构建 PyG Data 对象
                GCNData = DATA.Data(
                    smiles=smiles,
                    genes=torch.tensor(genes).unsqueeze(0),
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
                )
                GCNData.num_nodes = GCNData.atoms_features.size(0)
                num_bonds = GCNData.bonds_features.size(0)
                GCNData.bond_batch = torch.zeros(num_bonds, dtype=torch.long)
                num_3d_nodes = GCNData.h_3d.size(0)
                GCNData.batch_3d = torch.zeros(num_3d_nodes, dtype=torch.long)
                data_list.append(GCNData)

            except Exception as e:
                print(f"Warning: failed to process {smiles} at index {i}. Reason: {e}")
                continue

        print('Data preparation done! Saving to file...')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>')
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use

    PATTEN = re.compile(r'\[[^\]]+\]'
                        # only some B|C|N|O|P|S|F|Cl|Br|I atoms can omit square brackets
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a))

        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx=[]
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3  # 3 for <mask> !!!!!!
            else:
                a = self.s2i[s]
            l.append(a)
            
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTEN, s) is not None:
                atom_idx.append(i)
        if return_atom_idx:
            return l, atom_idx
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s[i]
                if c == '<eos>':
                    break
                s.append(c)
            smiles.append(''.join(s))

        return smiles


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


def smile_parse(smiles, tokenizer: Tokenizer):
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    smi = tokenizer.parse(smiles)
    return smi



    
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



if __name__ == "__main__":
	Smiles = []
	for dt_name in ["head100.csv"]:
	    df = pd.read_csv(dt_name)
	    Smiles += list(df['smiles'])
	
	Smiles = set(Smiles)
	
	atom_graph = {}
	bond_graph = {}
	smile_sequence = {}
	three_d_graph = {}
	
	for smile in Smiles:
	    atom_g, bond_g = create_atom_bond_graph(smile)
	    atom_graph[smile] = atom_g
	    bond_graph[smile] = bond_g
	
	    se = smiles_to_sequence(smile)
	    smile_sequence[smile] = se
	
	    three_d_g = smiles_to_egnn(smile)
	    three_d_graph[smile] = three_d_g
	
	# ------------------------
	# Step 2: 转换 CSV 数据为 PyTorch 格式
	# ------------------------
	dir = 'data'
	datasets = ["head100.csv"]
	
	for dataset in datasets:
	    processed_data = f'data/processed/{dataset.replace(".csv", "")}.pt'
	    tokenizer_file = f'{dir}/{dataset.replace(".csv", "")}_tokenizer.pkl'
	    print(processed_data)
	    if not os.path.isfile(processed_data):
	        df = pd.read_csv(dataset)
	
	        all_smiles = set(df['smiles']).union(set(df['smiles']))  # 这里 union(df['smiles']) 有点冗余，但保持原逻辑
	        tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))
	
	        # 保存 tokenizer
	        with open(tokenizer_file, 'wb') as file:
	            pickle.dump(tokenizer, file)
	
	        # Process train set
	        drugs = list(df['smiles'])
	        gene = list(df.drop('smiles', axis=1).to_numpy())
	
	        drugs = np.asarray(drugs)
	        gene = np.asarray(gene)
	
	        print('Preparing', dataset + '.pt in PyTorch format!')
	        train_data = TestbedDataset(
	            root='data',
	            dataset=dataset,
	            smi=drugs,
	            gene=gene,
	            atom_graph=atom_graph,
	            bond_graph=bond_graph,
	            smile_sequences=smile_sequence,
	            three_d_graph=three_d_graph
	        )
	
	        print(processed_data, 'has been created')
	    else:
	        print(processed_data, 'is already created')