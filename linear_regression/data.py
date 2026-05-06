from sklearn.model_selection import train_test_split
import numpy as np

def generate_data(n=250, noise=0.5):
    x = np.random.rand(n, 1)
    y = 2 + 6 * x + np.random.randn(n, 1) * noise # more noticeable noise
    return x, y

def prepare_data(x, y, test_size = 0.2, random_state = 42):
    x_train, x_test, y_train, y_test=train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    return x_train, x_test, y_train, y_test

"""
# The first column contains random values ​​from 0 to 1 (np.random.rand)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# add column of ones in train and test sets (bias term) 
x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

# The second column, units (np.ones), is the intercept for linear regression.
print(x_train.shape, x_test.shape)
"""
