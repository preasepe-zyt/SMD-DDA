import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, y):


        Q = self.q_proj(x)   # [N, d]
        K = self.k_proj(y)   # [M, d]
        V = self.v_proj(y)   # [M, d]

        attn = torch.matmul(Q, K.transpose(0, 1)) * self.scale   # [N, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [N, d]

        out = self.out_proj(out)

        return out
