'''
Quantifying a learning algorithm's performance and comparing different models are some of the
common tasks when applying machine learning to real world applications.
In this lab, you will practice doing these using the tips shared in class.
 Specifically, you will:

    split datasets into training, cross validation, and test sets
    evaluate regression and classification models
    add polynomial features to improve the performance of a linear regression model
    compare several neural network architectures

This lab will also help you become familiar with the code you'll see in this week's programming assignment. Let's begin!
'''
# for array computation and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for building and training neural networks
import tensorflow as tf

# load data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
            [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
            [21], [22], [23], [24], [25], [26], [27], [28], [29], [30],
            [31], [32], [33], [34], [35], [36], [37], [38], [39], [40],
            [41], [42], [43], [44], [45], [46], [47], [48], [49], [50]
              ])
Y = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
            [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
            [21], [22], [23], [24], [25], [26], [27], [28], [29], [30],
            [31], [32], [33], [34], [35], [36], [37], [38], [39], [40],
            [41], [42], [43], [44], [45], [46], [47], [48], [49], [50]
              ])

print(X.shape)
print(Y.shape)

# # Get 60% of the dataset as the training set.
# Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(X, Y, test_size=0.4, random_state=1)
# # Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
# Delete temporary variables
del x_, y_


print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

# perform feature scaling to help your model converge faster.
# This is especially true if your input features have widely different ranges of values.
scaler_linear = StandardScaler()
x_train_scaled = scaler_linear.fit_transform(x_train)
print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

# create and train a regression model. F
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)

# evaluate the performance of your model, you will measure the error for the training and cross validation sets.
# calculate mean squared error of training set
y_pred = linear_model.predict(x_train_scaled)

# use function in sklearn
err = mean_squared_error(y_pred, y_train) / 2
print(f"training MSE (using sklearn function): {err}")

# own for-loop implemenation
total_err = 0
for i in range(len(y_pred)):
    error = (y_pred[i] - y_train[i])**2
    total_err += error
mse = total_err / (2 * len(y_pred))
print(f"training MSE (for-loop implementation): {mse.squeeze()}")


# compute the MSE for the cross validation set
x_cv_scaled = scaler_linear.transform(x_cv)
y_pred_cv = linear_model.predict(x_cv_scaled)
err_cv = mean_squared_error(y_pred_cv, y_cv) / 2
print(f"validation MSE (using sklearn function): {err_cv}")



# Adding Polynomial Features
### Create the additional features
# First, you will generate the polynomial features from your training set.
# The code below demonstrates how to do this using the [`PolynomialFeatures`] class.
# It will create a new input feature which has the squared values of the input `x` (i.e. degree=2).
#
# Instantiate the class to make polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_polyed = poly.fit_transform(x_train)
print(x_train_polyed)

# scale the inputs as before to narrow down the range of values.
scaler_poly = StandardScaler()
x_train_polyed_scaled = scaler_poly.fit_transform(x_train_polyed)
print(x_train_polyed_scaled)

# train
model = LinearRegression()
model.fit(x_train_polyed_scaled, y_train)
y_pred_poly = model.predict(x_train_polyed_scaled)
print(f" Polyed Training MSE: {mean_squared_error(y_train, y_pred_poly) / 2}")

# validation
x_cv_polyed = poly.transform(x_cv)
x_cv_polyed_scaled = scaler_poly.transform(x_cv_polyed)
y_cv_pred_polyed_scaled = model.predict(x_cv_polyed_scaled)
print(f"Polyed Cross validation MSE: {mean_squared_error(y_cv_pred_polyed_scaled, y_cv) / 2}")



####  define multiple polynomial functions with different degree
# Initialize lists containing the lists, models, and scalers
train_mses = []
cv_mses = []
models = []
scalers = []

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for degree in range(1, 11):
    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)

    # Add polynomial features and scale the cross validation set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)


# Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")


### Add polynomial features to the test set with best performance
poly = PolynomialFeatures(degree, include_bias=False)
X_test_mapped = poly.fit_transform(x_test)

# Scale the test set
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")





# ### the same model selection process can also be used when choosing between different neural network architectures.
# Add polynomial features
degree = 1
poly = PolynomialFeatures(degree, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)

# Scale the features using the z-score
scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)


