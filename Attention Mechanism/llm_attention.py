""" 
To multiply matrices Q, K, and V, use the .matmul() method.
Transposition is performed using the .transpose method.
"""

import math, torch
import numpy as np

torch.manual_seed(42)

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1) # demension for the vector query
    # compute the attention scores by using torch.matmul
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # compute the result as the values weighted by attention propabilities (again, using torch.matmul)
    result = torch.matmul(p_attn, value)
    return result

query = torch.tensor([[0, 0], [0, 1], [1, 1]], dtype=torch.float)
key = torch.tensor([[100, 0], [0, 100], [0, 0]], dtype=torch.float)
value = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.float)
results = attention(query, key, value)
print(f"Query:\n{query}")
print(f"Key:\n{key}")
print(f"Value:\n{value}")
print(f"Results:\n{results}")

assert np.allclose(
    results[0].numpy(), [1 / 3, 1 / 3])
assert np.allclose(
    results[1].numpy(), [0,1])
    # the second query attends only to the second key
assert np.allclose(
    results[2].numpy(), [1 / 2, 1 / 2]
) # the third query attends to the first and second key equally


