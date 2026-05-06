import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 + 6 * x + (np.random.rand(100, 1) - 0.5)

plt.figure(figsize=(5, 3))
plt.scatter(x, y, s=10)
for w in [6.1, 5.8]:
    for b in [1.9, 2.3]:
        y_predicated = w * x + b
        plt.plot(x, y_predicated, color="r", alpha=0.3)
plt.plot(x, 2 + 6 * x, color="g")
plt.xlabel("x")
plt.ylabel("y")
plt.show()