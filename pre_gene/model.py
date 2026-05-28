from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import json
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv,global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from chemprop.models.mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
#EGNN
from .egnn import EGNN
#KAN
from .kan import KAN as EKAN
from .gated import Gated

class mlp(torch.nn.Module):  
    def __init__(self, num_features_xd=0, output_dim=1, dropout = 0):
        super(mlp, self).__init__()
        self.ffn1 = nn.Linear(978,512)
        self.ffn2 = nn.Linear(512,512)
        self.ffn3 = nn.Linear(512,512)
        self.ffn4 = nn.Linear(512, 978)
        self.dropout = nn.Dropout(0.2)
        self.relu    = nn.ReLU()

#     #---------------------------------------
    def forward(self, gene_mol):
        gene_mol = self.ffn1(gene_mol)
        gene_mol = torch.relu(gene_mol)
        gene_mol = self.dropout(gene_mol)
        gene_mol = self.ffn2(gene_mol)
        gene_mol = torch.relu(gene_mol)
        gene_mol = self.dropout(gene_mol)
        gene_mol = self.ffn3(gene_mol)
        gene_mol = torch.relu(gene_mol)
        gene_mol = self.dropout(gene_mol)
        gene_mol = self.ffn4(gene_mol)
        return gene_mol



class graph_encoder(nn.Module):
    def __init__(self, Drug_Features):
        super(graph_encoder, self).__init__()
        self.GraphConv1 = GCNConv(Drug_Features, Drug_Features * 2)
        self.GraphConv2 = GCNConv(Drug_Features * 2, Drug_Features * 3)
        self.GraphConv3 = GCNConv(Drug_Features * 3, Drug_Features * 4)

    def forward(self, atom, edge_index):
        GCNConv = self.GraphConv1(atom, edge_index)
        GCNConv = self.GraphConv2(GCNConv, edge_index)
        GCNConv  = self.GraphConv3(GCNConv, edge_index)

        return GCNConv


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 双向 GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True)
                          
    def forward(self, x):
        # Embedding
        embed = self.embedding(x) 
        # GRU
        gru_out, _ = self.gru(embed) 
        return gru_out



class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.loss_function = args.loss_function
        self.gene2id = None
        self.gene_ctp_list = None
        self.gene_ctp_ids = None

        self.seq =  RNNEncoder(1000, embed_dim = 128, hidden_dim = 256)
        self.atom_graph = graph_encoder(94)
        self.bond_graph = graph_encoder(4)
        self.three_d_graph = EGNN(in_node_nf=100, hidden_nf=100, out_node_nf=512, in_edge_nf=4, normalize=True, tanh=True)
        self.Gated = Gated(L=1719, D=512, dropout=0.35, out=978)

        
        self.extra_data_readin(args)
        self.gene_embedding_layer = nn.Embedding(len(self.gene_ctp_list),300)

        self.GCN1 = GCNConv(300, 300)
        self.GCN2 = GCNConv(300, 300)

        initialize_weights(self)
        self.device = args.device
        self.batch = args.batch_size


        #
        self.kan = EKAN(978, 978, 512, base_activation=torch.nn.ReLU)
        self.mlp = mlp()
        
    def extra_data_readin(self, args: TrainArgs) -> None:
        train = pd.read_csv('./pre_gene/train_full.csv', index_col=0)
        self.gene_ctp_list = list(train.columns) #
        self.gene2id = {gene: i for i, gene in enumerate(self.gene_ctp_list)}

        #construct gene coexpression network based on the training data
        correlation_matrix = train.corr(method='pearson')
        corr_threshold = 0.4
        edges = []
        # edge_weight = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > corr_threshold:
                    edges.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
                    # edge_weight.append(corr)
        edges = np.array(edges)

        self.gene2id = {gene: i for i, gene in enumerate(self.gene_ctp_list)}
        self.gene_ctp_ids = torch.LongTensor([self.gene2id[gene] for gene in self.gene_ctp_list]).to(args.device)

        edge_index = np.vectorize(self.gene2id.__getitem__)(edges)
        edge_index = torch.from_numpy(edge_index.T)
        edge_index = edge_index.to(torch.long).contiguous()
        self.edge_index = edge_index.to(args.device)
        

    def forward(self,data):
# sequence encoder 输出
        sequence = self.seq(data.seqs)
        sequence, _  = sequence.max(dim=1)
        
        atom = self.atom_graph(data.atoms_features, data.edge_index)
        atom = gmp( atom, data.batch)
        
        bond = self.bond_graph(data.bonds_features, data.bond_index)
        bond = gmp(bond, data.bond_batch)
        
        h, x = self.three_d_graph(data.h_3d, data.x_3d, data.edge_index_3d, data.edge_attr_3d)
        h = gmp(h, data.batch_3d)
        x = gmp(x, data.batch_3d)
        #print(h.shape,x.shape,sequence.shape, atom.shape, bond.shape)
        mol_f = torch.cat([h,x,sequence, atom, bond], dim=1)
        batch_size = mol_f.shape[0]

# gene embedding
        gene_embeddings = self.gene_embedding_layer(self.gene_ctp_ids)
        gene_embeddings =self.GCN1(gene_embeddings, edge_index=self.edge_index)
        gene_embeddings = self.GCN2(gene_embeddings, edge_index=self.edge_index)

# 扩展到 batch
        gene_embeddings = gene_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 978, 300)

# 扩展分子 embedding 到 gene 维度
        mol_f = mol_f.unsqueeze(1).expand(-1, gene_embeddings.size(1), -1)       # (B, 978, 300)

# 拼接   
        #print(gene_embeddings.shape, mol_f.shape)
        gene_mol = torch.cat([gene_embeddings, mol_f], dim=2)                     # (B, 978, 600)
        gene_mol = self.Gated(gene_mol)
        gene_mol = self.kan(gene_mol)  # KAN 接受 [B*N, F]
        #gene_mol = self.mlp(gene_mol) 
        
        output = torch.diagonal(gene_mol, dim1=-2, dim2=-1)
        return output

