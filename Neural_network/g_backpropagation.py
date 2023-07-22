'''
Gradient descent requires the derivative of the cost with respect to each parameter
in the network.
Neural networks can have millions or even billions of parameters.

The back propagation algorithm is used to compute those derivatives.
Computation graphs are used to simplify the operation.
'''

import numpy as np
from sympy import *
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets

w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")

# Backprop is the algorithm we use to calculate derivatives.
# As described in the lectures, backprop starts at the right and moves to the left.
# working right to left, for each node:
#
#     calculate the local derivative(s) of the node
#     using the chain rule, combine with the derivative of the cost with respect to the node to the right.


# Arithmetically
a_epsilon = a + 0.001       # a epsilon
J_epsilon = a_epsilon**2    # J_epsilon
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

# symbolically
sw,sJ,sa = symbols('w,J,a')
sJ = sa**2
sJ.subs([(sa,a)])
dJ_da = diff(sJ, sa)




