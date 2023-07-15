import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('./deeplearning.mplstyle')




# def compute_model_output(X, w, b):
#     m = X.shape[0]
#     y_hat = np.zeros(m)
#     for i in range(m):
#         y_hat[i] = X[i] * w + b
#
#     return y_hat


def compute_cost(y_train, X_train, w, b):
    sum_cost = 0
    m = X_train.shape[0]
    for i in range(m):
        cost = (X_train[i] * w + b - y_train[i]) ** 2
        sum_cost += cost

    sum_cost /= 2*m
    return sum_cost

def compute_derivative(X_train, y_train, w, b):
    m = X_train.shape[0]
    sum_d_w = 0
    sum_d_b = 0

    for i in range(m):
        d_w = (w * X_train[i] + b - y_train[i]) * X_train[i]
        sum_d_w += d_w
        d_b = w * X_train[i] + b - y_train[i]
        sum_d_b += d_b

    return sum_d_w/m, sum_d_b/m

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

X_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(X_train.shape)
# m = X_train.shape[0]

w = 0
b = 0
lr = 1.0e-2
print(lr)
steps = 10000

w, b, cost_hist, wb_hist = perform_gradient_descent(X_train, y_train, w, b, lr, steps, compute_cost, compute_derivative)

print(f"(w,b) found by gradient descent: ({w:8.4f},{b:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(cost_hist[:100])
ax2.plot(1000 + np.arange(len(cost_hist[1000:])), cost_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()


