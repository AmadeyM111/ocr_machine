# Gradient descent

import numpy as np
from sklearn.metrics import mean_squared_error

def gradient(x, y, w):
    """Gradient of mean squared error"""
    # Ensure y is column vector for proper broadcasting
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    grad = 2 * x.T @ (x @ w - y) / len(x)
    # Ensure gradient has shape (2, 1) to match w
    return grad.reshape(-1, 1) if grad.ndim == 1 else grad

def gradient_descent(x_train, y_train, x_test, y_test, w, alpha, iterations=100):
    """Gradient descent for optimization slope in simple linear regression"""
    # history
    w = w.copy()
    mse_train = []
    mse_test = []
    ws = [w.copy()]  # Store initial weights

    for i in range(iterations + 1):
        if i > 0:
            grad = gradient(x_train, y_train, w)
            w -= alpha * grad # adjust w based on gradient * learning rate
            # history
            mse_train.append(mean_squared_error(y_train, x_train @ w))
            mse_test.append(mean_squared_error(y_test, x_test @ w))
            ws.append(w.copy())  # Store weights at each iteration

            if i % 10 == 0 or i == iterations:
                print(f"Iter {i:4d} | b={w[0, 0]:.4f} w={w[1, 0]:.4f} "
                        f"train={mse_train[-1]:.6f} test={mse_test[-1]:.6f}")

    return w, mse_train, mse_test, ws
