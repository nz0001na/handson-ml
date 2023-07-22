'''
This lab will give you a more intuitive understanding of derivatives.
It will show you a simple way of calculating derivatives arithmetically.
It will also introduce you to a handy Python library that allows you to calculate derivatives symbolically.
'''

from sympy import symbols, diff

J = (3)**2
J_epsilon = (3+0.001)**2
k = (J_epsilon - J) / 0.001 # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")

J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

# Define the python variables and their symbolic names.
J, w = symbols('J, w')

# Define and print the expression.
J=w**2
print(J)

# Use SymPy's diff to differentiate the expression
# for 𝐽 with respect to 𝑤. Note the result matches our earlier example.
dJ_dw = diff(J,w)
print(dJ_dw)

# Evaluate the derivative at a few points by
# 'substituting' numeric values for the symbolic values.
# In the first example, 𝑤 is replaced by 2.

dJ_dw.subs([(w,2)])
dJ_dw.subs([(w,3)])
dJ_dw.subs([(w,-3)])

w, J = symbols('w, J')
J = 2 * w
dJ_dw = diff(J,w)
dJ_dw.subs([(w,-3)])

# Compare this with the arithmetic calculation
J = 2*3
J_epsilon = 2*(3 + 0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

J, w = symbols('J, w')
J=w**3
dJ_dw = diff(J,w)
dJ_dw.subs([(w,2)])

J = (2)**3
J_epsilon = (2+0.001)**3
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

J, w = symbols('J, w')
J= 1/w
dJ_dw = diff(J,w)
dJ_dw.subs([(w,2)])

J = 1/2
J_epsilon = 1/(2+0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")














