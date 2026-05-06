from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np


class PositionalFeedForward(nn.Module):
    def __init__(self, d.model, d_ff, dropout):
        super(PositionalWiseFeedForward, self). __init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Deopout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self, w_1(x))))
        