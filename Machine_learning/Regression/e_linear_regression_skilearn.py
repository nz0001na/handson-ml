'''
    This code is a practice of Gradient Descent Linear Regression model

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X_train, y_train = load_data()

# Scale/normalize the training data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

#  fit the model
sgd_r = SGDRegressor(max_iter = 1000)
sgd_r.fit(X_norm, y_train)
print(sgd_r)

# view parameters: bias, weight
b_norm = sgd_r.intercept_
w_norm = sgd_r.coef_

# make a prediction
y_pred_sgd = sgd_r.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm

# plot
fig, ax = plt.subplots(1, 4, figsize=(12,3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i], y_pred, color = dlc['dlorange'], label = 'predict')

ax[0].set_ylabel('price'); ax[0].legend();
fig.suptitle('target vs. prediction using z-score normalization model')
plt.show()

