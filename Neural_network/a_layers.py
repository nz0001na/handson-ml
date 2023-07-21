'''
Tensorflow is a machine learning package developed by Google.
In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0.
Keras is a framework developed independently by François Chollet that creates a
simple, layer-centric interface to Tensorflow.
This course will be using the Keras interface.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

# load data
X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

# Visualize the database
fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label='Data points')
ax.legend(fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

# create a layer
linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
# get weights and bias
w = linear_layer.get_weights()
print(w)

a1 = linear_layer(X_train[0].reshape(1,1))
w, b = linear_layer.get_weights()
print('w = {}, b = {}'.format(w, b))

# set fixed weights, bias
w_in = np.array([[200]])
b_in = np.array([100])
linear_layer.set_weights([w_in, b_in])
print(linear_layer.get_weights())

# compare layer and linear regression
a1 = linear_layer(X_train[0].reshape(1,1))
a1_ = np.dot(X_train[0].reshape(1,1), w_in) + b_in
print(a1)
print(a1_)

# predict
y_pred_ft = linear_layer(X_train)
y_pred_lr = np.dot(X_train, w_in) + b_in
print(y_pred_ft)
print(y_pred_lr)


# dataset
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
# get all index with label =1 / 0
pos = Y_train == 1
neg = Y_train == 0
X_train[pos]

fig, ax = plt.subplots(1,1, figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s = 80, c = 'r', label='y=1')
ax.scatter(X_train[neg], Y_train[neg], marker='x', s=100, c = 'b', label='y=0')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_ylim(-0.08, 1.1)
ax.set_title('One Variable Plot')
ax.legend(fontsize=12)
plt.show()

# create a model with logistic neuron
model = Sequential([
    tf.keras.layers.Dense(1, input_dim=1, activation='sigmoid', name = 'L1')
])
model.summary()

# get weights, bias of L1 layer
logis_layer = model.get_layer('L1')
w, b = logis_layer.get_weights()
print(w,b)
print(w.shape, b.shape)

# set weights, bias to L1 layer
w_in = np.array([[2]])
b_in = np.array([-4.5])
logis_layer.set_weights([w_in, b_in])
print(logis_layer.get_weights())

# compare output
y_pred_model = model.predict(X_train[0].reshape(1,1))
y_pred_alg = 1 / (1 + np.exp(-(np.dot(X_train[0], w_in) + b_in)))
print(y_pred_model)
print(y_pred_alg)








