from argparse import Namespace
from typing import List, Optional
import torch
from torch.nn import ModuleList, ReLU, Linear as Lin, Sequential as Seq, Sigmoid
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GINEConv,
    NNConv,
    GCNConv,
    PNAConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    Set2Set,
    Sequential,
    MLP
)
import torch.nn as nn
from egnn import EGNN
#KAN
from kan import KAN as EKAN
from gated import Gated
from torch_geometric.utils import to_dense_batch

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
import torch.nn.functional as F
class AttnPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttnPooling, self).__init__()
        self.attn_proj = nn.Linear(input_dim, 1)

    def forward(self, h, mask=None):

        # 1. raw attention score
        scores = self.attn_proj(h)  # [B, N, 1]

        # 2. squeeze last dim for softmax
        scores = scores.squeeze(-1)  # [B, N]

        # 3. masked softmax（关键）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=1)

        # 4. optional: safety (padding强制为0)
        if mask is not None:
            scores = scores * mask
            scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)

        # 5. restore shape
        scores = scores.unsqueeze(-1)  # [B, N, 1]

        return scores


class GNN(torch.nn.Module):
    def __init__(
        self
    ):
        super(GNN, self).__init__()
        self.atom_graph = graph_encoder(94)
        self.seq =  RNNEncoder(1000, embed_dim = 128, hidden_dim = 256)
        self.bond_graph = graph_encoder(4)
        self.three_d_graph = EGNN(in_node_nf=100, hidden_nf=100, out_node_nf=512, in_edge_nf=4, normalize=True, tanh=True)
        #self.Gated = Gated(L=1419, D=1024, dropout=0.30, out=512)
        self.Gated = Gated(L=1419, D=1024, dropout=0.10, out=512)
        self.kan = EKAN(512, 1, 512, base_activation=torch.nn.ReLU) #
        self.res = torch.nn.Linear(1419, 512)

       


        self.mlp = MLP([512, 512, 1], dropout=0.3, norm=None)
        self.att = AttnPooling(376)


    def forward(self, data, batch: Optional[torch.Tensor] = None):
        sequence = self.seq(data.seqs)
        sequence, _  = sequence.max(dim=1)
        
        atom = self.atom_graph(data.atoms_features, data.edge_index)
        atom = global_max_pool( atom, data.batch)

        atom_w = self.atom_graph(data.atoms_features, data.edge_index)
        graph, mask = to_dense_batch(atom_w, data.batch)
        weight = self.att(graph, mask)
        
        bond = self.bond_graph(data.bonds_features, data.bond_index)
        bond = global_max_pool(bond, data.bond_batch)
        
        h, x = self.three_d_graph(data.h_3d, data.x_3d, data.edge_index_3d, data.edge_attr_3d)
        h = global_max_pool(h, data.batch_3d)
        x = global_max_pool(x, data.batch_3d)
        mol_f = torch.cat([h,x,sequence, atom, bond], dim=1)
        #mol_f = self.Gated(mol_f)+self.res(mol_f)
        mol_f = self.Gated(mol_f)
        #x = self.mlp(mol_f)
        x = self.kan(mol_f)
        return x, weight.squeeze(-1)
    
    
