import copy
import math
import torch
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute scaled dot-product attention."
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in module size and number of heads"
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v_always equels d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]
        
        # Apply attention on all the projected vectors in batch
        x = attention(query, key, value, mask=mask, dropout=self.dropout)

        # "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

if __name__ == "__main__":
    torch.manual_seed(42)
    batch = 2
    seq_len = 4
    d_model = 8
    heads = 2
    mha = MultiHeadAttention(heads, d_model, dropout=0.0)
    query = torch.randn(batch, seq_len, d_model)
    key = torch.randn(batch, seq_len, d_model)
    value = torch.randn(batch, seq_len, d_model)
    out = mha(query, key, value)
    print("Input shape:", query.shape)
    print("Output shape:", out.shape)
    print("Output sample:\n", out[0])