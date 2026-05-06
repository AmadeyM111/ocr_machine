import numpy as np
from data import generate_data, prepare_data
from model import gradient_descent
from visualization import plot_mse

def main():
    # data generation
    x, y = generate_data(n=1000, noise=0.8)

    # data preparation
    x_train, x_test, y_train, y_test = prepare_data(x, y)

    # starting weights
    w_init = np.array([[0.], [0.]])

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # learning
    w, mse_train, mse_test = gradient_descent(
        x_train, y_train,
        x_test, y_test,
        w_init, 
        alpha=0.05, 
        iterations=800
    )

    print(f"\nFinal weights: bias = {w[0, 0]:.4f}, slope = {w[1, 0]:.4f}")

    # visualization
    plot_mse(mse_train, mse_test)

if __name__ == "__main__":
    main()
