import numpy as np
import matplotlib.pyplot as plt
import copy, math
np.set_printoptions(precision=2)


def compute_cost(y_train, X_train, w, b):
    sum_cost = 0
    m = X_train.shape[0]
    for i in range(m):
        cost = (np.dot(X_train[i], w) + b - y_train[i]) ** 2
        sum_cost += cost

    sum_cost /= 2*m
    return sum_cost

def compute_derivative(X_train, y_train, w, b):
    m, n = X_train.shape
    d_w = np.zeros((n,))
    d_b = 0.0

    for i in range(m):
        err = np.dot(w, X_train[i]) + b - y_train[i]
        for j in range(n):
            d_w[j] = d_w[j] + err * X_train[i,j]

        d_b = d_b + err

    d_w = d_w / m
    d_b = d_b / m

    return d_w, d_b


# def compute_gradient(X, y, w, b):
#     m, n = X.shape  # (number of examples, number of features)
#     dj_dw = np.zeros((n,))
#     dj_db = 0.
#
#     for i in range(m):
#         err = (np.dot(X[i], w) + b) - y[i]
#         for j in range(n):
#             dj_dw[j] = dj_dw[j] + err * X[i, j]
#         dj_db = dj_db + err
#     dj_dw = dj_dw / m
#     dj_db = dj_db / m
#
#     return dj_db, dj_dw



def perform_gradient_descent(X_train, y_train, w_0, b_0, lr, steps, cost_func, derivative):
    w = w_0
    b = b_0
    cost_hist = []
    wb_hist = []

    cost = cost_func(y_train, X_train, w, b)
    cost_hist.append(cost)
    wb_hist.append([w, b])

    for n in range(steps):
        d_w, d_b = derivative(X_train, y_train, w, b)
        w = w - lr * d_w
        b = b - lr * d_b
        cost = cost_func(y_train, X_train, w, b)
        cost_hist.append(cost)
        wb_hist.append([w, b])
    return w, b, cost_hist, wb_hist


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
print(X_train.shape)
print(y_train.shape)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
lr = 5.0e-7
print(lr)
steps = 10

w, b, cost_hist, wb_hist = perform_gradient_descent(X_train, y_train, w_init, b_init, lr, steps, compute_cost, compute_derivative)

# print(f"(w,b) found by gradient descent: ({w:8.4f},{b:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(cost_hist[:100])
ax2.plot(1000 + np.arange(len(cost_hist[1000:])), cost_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()


