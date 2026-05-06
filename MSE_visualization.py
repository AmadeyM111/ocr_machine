import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 + 6 * x + (np.random.rand(100, 1) - 0.5)

def plot_delta_line(ax, x, y, w, b, color="r"):
    y_predicated = w * x + b
    # line
    ax.plot(x, y_predicated, color=color, alpha=0.5, label=f"f(x)={w}x+{b}")
    # delta line
    for x_i, y_i, f_x in zip(x, y, y_predicated):
        ax.vlines(x=x_i, ymin=min(f_x, y_i), ymax=max(f_x, y_i), ls="--", alpha=0.3)
        # MSE
    loss = np.sum((y - (w * x + b))**2) / (len(x))
    ax.set_title(f"MSE = {loss:.3f}")
    ax.legend()

fig, axs = plt.subplots(1, 2, figsize=(11, 4))

# plot x_i, y_i (dots)
for ax in axs:
    ax.scatter(x, y, s=10)
    ax.set_xlim([0, 0.8])
    ax.set_ylim([2, 6])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plot_delta_line(axs[0], x, y, w=5, b=2, color="r")
plot_delta_line(axs[1], x, y, w=6, b=2, color="g")

plt.show()