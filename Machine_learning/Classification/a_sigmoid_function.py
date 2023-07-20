import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    g = 1/ (1 + np.exp(-z))
    return g

# use np.exp() to calculate exponential values of array or scaler
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)
print(input_array)
print(exp_array)

input_value = 1
exp_value = np.exp(input_value)
print(input_value)
print(exp_value)

# calculate sigmoid function
z_value = np.arange(-10, 11)
y = sigmoid_function(z_value)
print(np.c_[z_value, y])

# plot sigmoid function
fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.plot(z_value, y, c='b')
ax.set_title('Sigmoid Function')
ax.set_ylabel('Sigmoid(z)')
ax.set_xlabel('z')
plt.show()