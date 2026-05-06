import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 + 6 * x + (np.random.rand(100, 1) - 0.5)

plt.figure(figsize=(5, 3))
plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()