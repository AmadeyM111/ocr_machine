#pip install -q torcheval

import torcheval
from torcheval.metrics import blue_score

blue = blue_score(pred_tokens_for_blue, eng)
print(f"BLUE = {blue.item():.2f}")

for n in range(2, 5):
    blue = blue score(pred_tokens_for_blue, eng, n_gram=n)
    print(f"BLUE ({n}-gram) = {blue.item():.2f}")