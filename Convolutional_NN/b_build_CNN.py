'''
Convolutional Neural Networks: Step by Step

Welcome to Course 4's first assignment! In this assignment, you will implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation.

By the end of this notebook, you'll be able to:

    Explain the convolution operation
    Apply two different types of pooling operation
    Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
    Build a convolutional neural network


'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# np.random.seed(1) is used to keep all the random function calls consistent.
np.random.seed(1)


# GRADED FUNCTION: zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))

    return X_pad


# GRADED FUNCTION: conv_single_step
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # Sum over all entries of the volume s.
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z


# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer,
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # 1. Retrieve dimensions from A_prev's shape (≈1 line)
    # (m, n_H_prev, n_W_prev, n_C_prev) = None
    # 2. Retrieve dimensions from W's shape (≈1 line)
    # (f, f, n_C_prev, n_C) = None
    # 3. Retrieve information from "hparameters" (≈2 lines)
    # stride = None
    # pad = None
    # 4. Compute the dimensions of the CONV output volume using the formula given above.
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    # n_H = None
    # n_W = None
    # 5. Initialize the output volume Z with zeros. (≈1 line)
    # Z = None
    # 6. Create A_prev_pad by padding A_prev
    # A_prev_pad = None

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1

    print('A_prev shape :', A_prev.shape)
    print('W.shape :', W.shape)
    print('stride :', stride)
    print('pad :', pad)
    print('n_H :', n_H)
    print('n_W :', n_W)

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    print('A_prev_pad shape :', A_prev_pad.shape)
    print('Z shape :', Z.shape)

    # loop over the batch of training examples
    for i in range(m):
        # Select ith training example's padded activation
        a_prev_pad = A_prev_pad[i]

        # loop over vertical axis of the output volume
        for h in range(0, n_H_prev + 2 * pad - f + 1, stride):
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h
            vert_end = h + f

            # loop over horizontal axis of the output volume
            for w in range(0, n_W_prev + 2 * pad - f + 1, stride):
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = w
                horiz_end = w + f

                # loop over channels (= #filters) of the output volume
                for c in range(n_C):
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    new_h = int(h / stride)
                    new_w = int(w / stride)
                    Z[i, new_h, new_w, c] = conv_single_step(a_slice_prev, weights, biases)


    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


'''
The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are:

    Max-pooling layer: slides an (𝑓,𝑓) window over the input and stores the max value of the window in the output.
    Average-pooling layer: slides an (𝑓,𝑓) window over the input and stores the average value of the window in the output.
    
These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the 
window size 𝑓. This specifies the height and width of the 𝑓×𝑓 window you would compute a max or average over. 

'''


# GRADED FUNCTION: pool_forward
def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        for h in range(0, n_H_prev - f + 1, stride):  # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h
            vert_end = h + f

            for w in range(0, n_W_prev - f + 1, stride):  # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                horiz_start = w
                horiz_end = w + f

                for c in range(n_C_prev):  # loop over the channels of the output volume

                    # Use the corners to define the current slice on the ith training example
                    #                     of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    new_h = int(h / stride)
                    new_w = int(w / stride)

                    # Compute the pooling operation on the slice.
                    # Use an if statement to differentiate the modes.
                    # Use np.max and np.mean.
                    if mode == "max":
                        A[i, new_h, new_w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, new_h, new_w, c] = np.mean(a_prev_slice)

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    # assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache










