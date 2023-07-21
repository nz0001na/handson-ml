'''
we will build a small neural network using Numpy.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# define sigmoid function
def sigmoid_function(z):
    g = 1 / (1 + np.exp(-z))
    return g

# define my own dense function
def my_dense_vectorization(a_in, W, b):
    """
        Computes dense layer
        Args:
          a_in (ndarray (n, )) : Data, 1 example
          W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
          b    (ndarray (j, )) : bias vector, j units
        Returns
          a_out (ndarray (j,))  : j units|
        """
    a_out = sigmoid_function(np.matmul(a_in, W) + b)
    return a_out


# define my own sequential model with 2-layer
def my_sequential(x, W1, b1, W2, b2):
    """
       Computes sequential output
       Args:
         x     (ndarray (n, )) : Data, 1 example
         W1    (ndarray (n,j)) : Layer 1 Weight matrix, n features per unit, j units
         b1    (ndarray (j, )) : Layer 1 bias vector, j units
         W2    (ndarray (n,j)) : Layer 2 Weight matrix, n features per unit, j units
         b2    (ndarray (j, )) : Layer 2 bias vector, j units
       Returns
         f_x   scaler, propability
       """
    a1 = my_dense_vectorization(x, W1, b1)
    a2 = my_dense_vectorization(a1, W2, b2)
    f_x = a2
    return f_x


# define my own predict function
def my_predict(X, W1, b1, W2, b2):
    """
       make a prediction
       Args:
         X     (ndarray (m, n)) : Data, m examples
         W1    (ndarray (n,j)) : Layer 1 Weight matrix, n features per unit, j units
         b1    (ndarray (j, )) : Layer 1 bias vector, j units
         W2    (ndarray (n,j)) : Layer 2 Weight matrix, n features per unit, j units
         b2    (ndarray (j, )) : Layer 2 bias vector, j units
       Returns
         p     (ndarray (m, )):  propability
       """
    m = X.shape[0]
    p = np.zeros((m,))
    for i in range(m):
        x = X[i]
        p[i] = my_sequential(x, W1, b1, W2, b2)

    return p




# load data
X_train = np.array([[30, 23],
                    [40, 39],
                    [90, 78],
                    [12, 12]
                    ])
y_train = np.array([0, 1, 1, 0])
print(X_train.shape, y_train.shape)

# Normalize the data
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X_train)
Xn = norm_l(X_train)

# Set weights, bias: 2x3, 1x3 of layer 1
W1_ini = np.array([
    [-8.93, 0.29, 12.9],
    [-0.1, -7.32, 10.81]
])
b1_ini = np.array([-9.82, -9.28, 0.96])
# set weights, bias of layer 2: 3x1, 1x1
W2_ini = np.array([
    [-31.18],
    [-27.59],
    [-32.56]
])
b2_ini = np.array([15.41])

# make prediction
X_test = np.array([
     [200, 13.9],
     [200, 17]
])
X_t = norm_l(X_test)
y_pred = my_predict(X_t, W1_ini, b1_ini, W2_ini, b2_ini)
y_hat = (y_pred >= 0.5).astype(int)
print(y_pred)
print(y_hat)

