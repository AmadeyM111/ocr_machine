class Encoder(nn.Module):
    "Core encoder is a stuck of N layers"

def __init(self, layer, N):
    super(Encoder, self).__init__()
    self.layers = clone(layer, N)
    self.norm = LayerNorm(layer.size)

def forward(self, x, mask):
    "Pass the input (and mask) though each layer in turn"
    for layer in self.layers:
        x = layer(x, mask)
    return self.norm(x)
