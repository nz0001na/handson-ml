'''
Softmax Regression and in Neural Networks when solving Multiclass Classification problems.

Notes: SparseCategorialCrossentropy or CategoricalCrossEntropy

    Tensorflow has two potential formats for target values and the selection of the loss defines which
    is expected.

    (1) SparseCategorialCrossentropy:
    expects the target to be an integer corresponding to the index. For example, if there are 10
    potential target values, y would be between 0 and 9.

    (2) CategoricalCrossEntropy:
    Expects the target value of an example to be one-hot encoded where the value at
    the target index is 1 while the other N-1 entries are zero. An example with 10 potential
    target values, where the target is 2 would be [0,0,1,0,0,0,0,0,0,0].


'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def my_softmax(z):
    e_z = np.exp(z)
    soft_max = e_z / np.sum(e_z)

    return soft_max

# data
# make datasets for example
centers = [[-5,2], [-2,-2], [1,2], [5,-2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)

# define a model
# (1) a straightforward way
model = Sequential([
    Dense(units=25, activation='relu', name='L1'),
    Dense(units=15, activation='relu', name='L2'),
    Dense(units=4, activation='softmax', name='L3')   # < softmax activation here
])

model.compile(
    loss=SparseCategoricalCrossentropy(),
    # loss=CategoricalCrossentropy(),
    optimizer=Adam(0.001),
)

model.fit(X_train, y_train, epochs=10)
# The output predictions are probabilities!
y_pred = model.predict(X_train)
print(y_pred[:5])
print('Max: {}'.format(np.max(y_pred)))
print('Min: {}'.format(np.min(y_pred)))



# (2) a stable way: preferred
prefer_model = Sequential([
    Dense(units=25, activation='relu', name='L1'),
    Dense(units=15, activation='relu', name='L2'),
    Dense(units=4, activation='linear', name='L3')
])
prefer_model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    # loss=CategoricalCrossentropy(from_logis=True),
    optimizer= Adam(0.001)
)
prefer_model.fit(X_train, y_train, epochs=10)
# The output predictions are not probabilities!
# If the desired output are probabilities, the output should be processed by a softmax.
y_pred_prefer = prefer_model.predict(X_train)
print('Output:')
print(y_pred_prefer[:5])
print('Max: {}'.format(np.max(y_pred_prefer)))
print('Min: {}'.format(np.min(y_pred_prefer)))


y_p = tf.nn.softmax(y_pred_prefer).numpy()
print('Probability on each class:')
print(y_p[:5])
print('Max: {}'.format(np.max(y_p)))
print('Min: {}'.format(np.min(y_p)))

for i in range(5):
    print( f"{y_p[i]}, category: {np.argmax(y_p[i])}")
