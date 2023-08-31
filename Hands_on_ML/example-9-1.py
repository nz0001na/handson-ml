'''
Linear Regression
mini-batch gradient decent
save model
adding timestamp in the log directory

'''

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime
# from tensorflow_graph_in_jupyter import show_graph
# show_graph(tf.get_default_graph())

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
# root_logdir = 'D:/000_machine_learning/hands-on-code/tf_logs'
root_logdir = 'C:/tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)


# data
housing = fetch_california_housing()
m, n = housing.data.shape
print(m, n)
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]

# placeholder
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch

# model
# model_path = 'D:/000_machine_learning/hands-on-code/e9/'
model_path = './e9/'
n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_predict = tf.matmul(X, theta, name='predictions')
with tf.name_scope('loss') as scope:
    error = y_predict - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:  # not shown
            save_path = saver.save(sess, model_path + "my_model.ckpt")
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    save_path = saver.save(sess, model_path + "my_model_final.ckpt")
    print(best_theta)

file_writer.close()
