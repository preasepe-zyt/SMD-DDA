import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Gated(nn.Module):
    def __init__(self, L=658, D=512, dropout=0.25, out=658):
        super(Gated, self).__init__()
        self.a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.a.append(nn.Dropout(dropout))
            self.b.append(nn.Dropout(dropout))

        self.a = nn.Sequential(*self.a)
        self.b = nn.Sequential(*self.b)

        self.c = nn.Linear(D, out)
        self.layer_norm = nn.LayerNorm(out)



    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        A = a.mul(b)
        A = self.c(A)
        A = self.layer_norm(A)
#        if x.shape[-1] == A.shape[-1]:
#            A = A + x
        return A
