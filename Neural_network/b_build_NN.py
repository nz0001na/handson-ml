'''
will build a small neural network using Tensorflow.
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

X_train = np.array([[30, 23],
                    [40, 39],
                    [90, 78],
                    [12, 12]
                    ])
y_train = np.array([0, 1, 1, 0])
print(X_train.shape, y_train.shape)

# Normalize Data
norm_l = Normalization(axis=-1)
norm_l.adapt(X_train)
Xn = norm_l(X_train)
print(Xn)
print(y_train)

# Tile/copy our data to increase the training set size and reduce the number of training epochs.
Xt = np.tile(Xn, (1000,1))
Yt = np.tile(y_train, (1000))
print(Xt.shape, Yt.shape)

# create a model
'''
Note 1: The tf.keras.Input(shape=(2,)), specifies the expected shape of the input.
    This allows Tensorflow to size the weights and bias parameters at this point. 
    This is useful when exploring Tensorflow models. 
    This statement can be omitted in practice and Tensorflow will size the network parameters 
    when the input data is specified in the model.fit statement.
Note 2: Including the sigmoid activation in the final layer is not considered 
    best practice. It would instead be accounted for in the loss which improves
     numerical stability. This will be described in more detail in a later lab.
'''
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(units=3, activation='sigmoid', name='L1'),
    Dense(units=1, activation='sigmoid', name='L2')
])
model.summary()
w1, b1 = model.get_layer('L1').get_weights()
w2, b2 = model.get_layer('L2').get_weights()
print(w1, b1)
print(w2, b2)

'''
The model.compile statement:
    defines a loss function and specifies a compile optimization.
The model.fit statement:
    runs gradient descent and fits the weights to the data.
'''
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# For efficiency, the training data set is broken into 'batches'. The default size of a batch in Tensorflow is 32.
model.fit(Xt, Yt, epochs=10)

# check updated weights, bias
w1, b1 = model.get_layer('L1').get_weights()
w2, b2 = model.get_layer('L2').get_weights()
print(w1, b1)
print(w2, b2)

# set new weights, bias
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]])
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("L1").set_weights([W1,b1])
model.get_layer("L2").set_weights([W2,b2])

# test
X_test = np.array([[200, 13.9],
                   [200, 17]])
X_n = norm_l(X_test)
y_pred = model.predict(X_n)
print(y_pred)

# classify
yhat = np.zeros_like(y_pred)
for i in range(len(y_pred)):
    if y_pred[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

# or
yhat = (y_pred >= 0.5).astype(int)
print(f"decisions = \n{yhat}")