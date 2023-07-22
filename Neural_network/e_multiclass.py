'''
 explore an example of multi-class classification using neural networks.
 a multiclass network in Tensorflow.
 4-class, 2-layer
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs

# load data
centers = [[-5,2], [-2,-2], [1,2], [5, -2]]
classes = 4
m = 100
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=30)

print('Unique classes: {}'.format(np.unique(y_train)))
print(y_train[:10])
print("X shape: ", X_train.shape)
print('y shape: ', y_train.shape)

# define model
# 2-layer, 4 outputs, one for each class
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential([
    Dense(units=2, activation='relu', name = 'L1'),
    Dense(units=4, activation='linear', name='L2')
])
model.compile(
    # Setting from_logits=True as an argument to the loss
    # function specifies that the output activation was linear rather than a softmax.
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01)
)
model.fit(X_train, y_train, epochs=10)
y_pred = model.predict(X_train)
print(y_pred[:10])
y_p = tf.nn.softmax(y_pred).numpy()
print(y_p[:10])

# get weights
l1 = model.layers[0]
l_1 = model.get_layer('L1')
print(l1.get_weights())
print(l_1.get_weights())












