'''

    extend the previous linear and logistic cost functions with a regularization term.
    rerun the previous example of over-fitting with a regularization term added.

'''

import numpy as np
import matplotlib.pyplot as plt


# calculate cost of linear regularization with regularization term
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """
    m,n = X.shape

    org_cost = 0
    for i in range (m):
        f = np.dot(X[i], w) + b
        err = (f - y[i])**2
        org_cost +=  err

    reg_term = 0
    for j in range(n):
        reg_term += w[j] ** 2

    total_cost = org_cost / (2*m) + lambda_ * reg_term/ (2*m)
    return total_cost


def sigmoid_function(z):
    g = 1 / (1 + np.exp(-z))
    return g

# calculate cost of logistic regression with regularization term
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """
    m,n = X.shape
    org_cost, reg_cost = 0, 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f = sigmoid_function(z)
        org_cost += - y[i] * np.log(f) - (1 - y[i]) * np.log(1 - f)

    org_cost /= m

    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = lambda_ * reg_cost/(2*m)

    total_cost = org_cost + reg_cost
    return total_cost


# calculate Gradient Descent linear regression with regularization term
def compute_gradient_linear_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        f = np.dot(X[i], w) + b
        err = f - y[i]
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i][j]

    dj_dw = dj_dw / m
    dj_db /= m

    for j in range(n):
        dj_dw[j] += lambda_ * w[j] / m

    return dj_dw, dj_db

# calculate Gradient Descent of Logistic regression with regularization term
def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f = sigmoid_function(z)
        err = f - y[i]
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i][j]

    dj_db /= m
    dj_dw /= m

    for j in range(n):
        dj_dw[j] += lambda_ * w[j] / m

    return dj_dw, dj_db



#  data: cost
np.random.seed(1)
X_train = np.random.rand(5,6)
y_train = np.array([0,1,0,1,0])



w_out = np.random.rand(X_train.shape[1]).reshape(-1,) - 0.5
b_out = 0.5
lambda_out = 0.7
lr_cost = compute_cost_linear_reg(X_train, y_train, w_out, b_out, lambda_out)
print(lr_cost)

logr_cost = compute_cost_logistic_reg(X_train, y_train, w_out, b_out, lambda_out)
print(logr_cost)


# data: GD
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

dj_db_tmpl, dj_dw_tmpl = compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"dj_db: {dj_db_tmpl}", )
print(f"Regularized dj_dw:\n {dj_dw_tmpl.tolist()}", )


