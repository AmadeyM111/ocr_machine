from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np


class PositionalEncodding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncodding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position / div_term) # definition of even grades
        pe[:, 1::2] = torch.cos(position / div_term) # definition of odd grades 
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False) # interpretation of the meaning of the word and its position
        return self.dropout(x)

import matplotlib.pyplot as plt 

plt.figure(figsize=(15, 5))
pe = PositionalEncodding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])

plt.show()