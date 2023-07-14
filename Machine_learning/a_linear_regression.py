import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')

w = 100
b = 100

def compute_model_output(X, w, b):
    m = X.shape[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = X[i] * w + b

    return y_hat




X_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(X_train.shape)
m = X_train.shape[0]

y_pred = compute_model_output(X_train, w, b)

plt.plot(X_train, y_pred, label = 'Predicted Values')
plt.scatter(X_train, y_train, marker = 'x', c = 'r', label = 'Actual Values')
plt.title('Housing Prices')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()


