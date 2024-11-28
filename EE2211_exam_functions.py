"""Functions needed for EE2211 exams  

NOTE: Assumes that the necessary python packages for EE2211
have been installed.  

"""

# Libraries needed for all the functions
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

"""----------Chapter 3 - 5----------
- bayes_theorem()
- check_rank()
"""

def bayes(A, B, B_given_A):
    """Compute the value of A|B (A given B) using Bayes theorem
    
    :param  
    A: the value of A from 0 to 1  
    B: the value of B from 0 to 1  
    B_given_A: the value of B|A from 0 to 1  
    :return  
    A_given_B: the value of A|B
    """
    A_given_B = (A*B_given_A)/B
    print("A given B is: ")
    print(A_given_B)

def check_rank(X):
    """Check the rank of a matrix X
    
    :param  
    X: the matrix X  
    :return  
    rank: the rank of X
    """
    rank = np.linalg.matrix_rank(X)
    print(f"Rank of X is: {rank}")
    return rank

def is_invertible(matrix):
    """Checks if a matrix is invertible.

    param:
    matrix (numpy.ndarray): A square matrix to check.
    return:
    bool: True if the matrix is invertible, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("The matrix is not square.")
        return False

    det = np.linalg.det(matrix) # find determinant
    if np.isclose(det, 0):
        print("The matrix is singular (det = 0) and not invertible.")
        return False
    else:
        print("The matrix is invertible.")
        return True

"""----------Chapter 6----------
- mean_squared_error()
- add_bias()
- make_poly()
- least_squares_prime()
- binary_classif()
- multi_category_classif()
- ridge_reg()
"""

def mean_squared_error(X, y):
    """calculates the MSE for a matrix X and target output vector y  
    
    :param  
    X: a matrix X  
    y: vector y  
    :return  
    mse: the mean squared error
    """
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict target values using the model
    y_pred = model.predict(X)

    # Calculate the MSE
    mse = np.mean((y - y_pred) ** 2)

    print(f"Mean squared error is: {mse}")
    return mse

def add_bias(X):
    """Adds the column of 1s as offset to a matrix  

    :param  
    X: a matrix X  
    :return  
    X: X with the bias column added
    """
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X

def make_poly(X, order):
    """Use this to convert X into a polynomial of given order.
    NOTE: REMEMBER TO MAKE BOTH X AND X_TEST POLY  
    
    :param  
    X: matrix X  
    order: desired order  
    :return  
    X: matrix transformed to have desired order
    """
    poly = PolynomialFeatures(order)
    return poly.fit_transform(X)

def least_squares_prime(X, y, Xt):
    """Predict a value of y using least squares regression.  
    
    :param  
    X: a matrix X  
    y: a target output vector y  
    Xt: the test X values  
    :return  
    y_out: the predicted y values    
    """ 
    # calculate w
    w = inv(X.T @ X) @ X.T @ y
    print(f"w is: {w}")

    # predict 
    y_out = Xt @ w
    print(f"predicted y is: {y_out}")
    return y_out
    
def binary_classif(X, y, Xt):
    """Predict a binary value of y (-1 or +1)  
    
    :param  
    X: a matrix X  
    y: a target output vector y as -1 or 1  
    Xt: the test X values  
    :return  
    y_out: the predicted y values as -1 or 1    
    """
    w = inv(X.T @ X) @ X.T @ y
    y_predict = Xt @ w
    #y_class_predict = np.sign(y_predict)
    y_class_predict = [[1 if x >= 0 else -1] for x in y_predict]
    print(f"Predicted y class: {y_class_predict}")
    return y_class_predict

def multi_category_classif(X, y, Xt):
    """Predict a value of multi-category y using multi-category classification.
    Does the conversion to one-hot values for y.
    
    :param  
    X: a matrix X  
    y: a target output vector y as integers (not one hot)
    Xt: the test X values  
    :return  
    y_out: the predicted y values in a one-hot matrix 
    """
    # one hot encode, find W
    encoder = OneHotEncoder(sparse=False)
    Ytr = encoder.fit_transform(y)
    W = inv(X.T @ X) @ X.T @ Ytr
    print(f"W is: {W}")

    # predict y
    yt = Xt @ W;
    yt_class = [[1 if y == max(x) else 0 for y in x] for x in yt ] 
    print(f"predicted y is: {yt_class}")
    return yt_class

def ridge_reg(X, y, reg_factor, Xt):
    """Predict a value of y using ridge regression.  

    :param  
    X: a matrix X  
    y: a target output vector y  
    reg_factor: the regularisation factor (lambda)  
    Xt: the test X values  
    :return  
    y_out: the predicted y values
    """

    # find w
    if (X.shape[0] > X.shape[1]):
        # primal (m > d)
        reg_L = reg_factor*np.identity(X.shape[1])
        w = inv(X.T @ X + reg_L) @ X.T @ y
    else:
        # dual (m <= d)
        reg_L = reg_factor*np.identity(X.shape[0])
        w = X.T @ inv(X @ X.T + reg_L) @ y
    
    # predict y
    yt = Xt @ w
    print(f"predicted y is: {yt}")
    return yt
       
"""----------Chapter 7----------

- pearsons() # Recommend just using microsoft excel for this tbh
- bias_squared()
- variance()
"""

def pearsons(x, y):
    """
    Calculate Pearson's correlation coefficient between two arrays.

    :param  
    x: array-like, feature values  
    y: array-like, target values  
    :return  
    r: Pearson's correlation coefficient

    Example arrays:
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Compute means
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Compute numerator (covariance)
    covariance = np.sum((x - mean_x) * (y - mean_y))

    # Compute denominator (product of standard deviations)
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))
    denominator = std_x * std_y

    # Avoid division by zero
    if denominator == 0:
        raise ValueError("Division by zero: standard deviation of x or y is zero.")

    # Pearson's correlation coefficient
    r = covariance / denominator
    return r

def bias_squared(y_true, y_predictions):
    """
    Calculate the Bias^2 of the predictions.

    param:  
    y_true (float): The true value.
    y_predictions (numpy.ndarray): An array of predicted values.  
    return:
    bias_sq: the bias^2 as a float
    """
    y_mean = np.mean(y_predictions)  # Mean of predictions
    bias_sq = (y_mean - y_true) ** 2
    return bias_sq

def variance(y_predictions):
    """
    Calculate the variance of the predictions.

    param:  
    y_predictions (numpy.ndarray): An array of predicted values.  
    return:  
    var: variance as a float
    """
    y_mean = np.mean(y_predictions)  # Mean of predictions
    var = np.mean((y_predictions - y_mean) ** 2)
    return var

"""----------Chapter 8----------

- grad_descent()  
- grad_descent_2()
"""

def grad_descent(x, eta, iterations, function, grad):
    """ Returns arrays of x and f(x) after specified number of iterations

    Args:
        x (float): the value of the x variable
        eta (float): the learning rate value
        iterations (int): the number of iterations
        function (function): the defined function for f(x), e.g. np.sin(x)**2
        grad (function): the defined function for f'(x), e.g. 2*np.sin(x)*np.cos(x)

    Returns:
        x_out (np.array): a numpy array of x values AFTER grad descent
        f_out (np.array): a numpy array of f(x) results AFTER grad descent
    """
    x_out = np.zeros(iterations)
    f_out = np.zeros(iterations)
    for i in range(iterations):
        x = x - eta * grad(x)
        x_out[i] = x
        f_out[i] = function(x)
    print("Values of x are: ", x_out)
    print("Function results are: ", f_out)
    return x_out, f_out

def grad_descent_2(x, y, eta, iterations, function, grad_x, grad_y):
    """ Returns arrays of x and f(x) after specified number of iterations
        #NOTE: when the gradient has the other variable, we just use the original value for the gradient calculation,
        i.e. we do not update it after the iterations

    param:  
        x (float): the value of the x variable
        y (float): the value of the y variable
        eta (float): the learning rate value
        iterations (int): the number of iterations
        function (function): the defined function for f(x, y)
        grad_x (function): the defined function for grad w.r.t x, i.e. f'_x(x)
        grad_y (function): the defined function for grad w.r.t y, i.e. f'_y(y)

    return:  
        x_out (np.array): a numpy array of x values AFTER grad descent
        y_out (np.array): a numpy array of y values AFTER grad descent
        f_out (np.array): a numpy array of f(x) results AFTER grad descent
    """
    x_out = np.zeros(iterations)
    y_out = np.zeros(iterations)
    f_out = np.zeros(iterations)
    for i in range(iterations):
        x = x - eta * grad_x(x)
        y = y - eta * grad_y(y)
        x_out[i] = x
        y_out[i] = y
        f_out[i] = function(x, y)
    print("Values of x are: ", x_out)
    print("Values of y are: ", y_out)
    print("Function results are: ", f_out)
    return x_out, y_out, f_out

"""----------Chapter 9----------

- gini_impurity()
- entropy()
- misclassification_rate()
- overall_metric() 
- split_data()
- mse_node()
- mse_depth_1()
"""

def gini_impurity(proportions):
    """
    Calculate the Gini impurity for a node.
    :param proportions: List or array of class proportions in the node
    :return: Gini impurity value

    Example proportions
    .
    [0.8, 0.2]
    """
    gini = 1 - np.sum(np.array(proportions) ** 2)
    print("Gini is: ", gini)
    return gini

def entropy(proportions):
    """
    Calculate the entropy for a node.
    :param proportions: List or array of class proportions in the node
    :return: Entropy value
    """
    proportions = np.array(proportions)
    proportions = proportions[proportions > 0]  # Avoid log(0)
    entropy = -np.sum(proportions * np.log2(proportions))
    print("Entropy is: ", entropy)
    return entropy 

def misclassification_rate(proportions):
    """
    Calculate the misclassification rate for a node.
    :param proportions: List or array of class proportions in the node
    :return: Misclassification rate
    """
    misclassification = 1 - np.max(proportions)
    print("Misclassification rate is: ", misclassification)
    return misclassification

def overall_metric(child_nodes, metric_function):
    """
    Calculate the overall metric (gini, entropy or misclassification rate) at a certain depth.
    :param child_nodes: List of tuples, where each tuple contains:
                        (number of samples in the node, class proportions in the node)
    :param metric_function: Function to calculate the metric for a single node (entropy or misclassification rate)
    :return: Overall metric value

    Example child nodes
    .
    Node 1: 40 samples, 80% Class A, 20% Class B
    Node 2: 60 samples, 50% Class A, 50% Class B
    child_nodes = [
        (40, [0.8, 0.2]),  # Node 1
        (60, [0.5, 0.5])   # Node 2
    ]
    """
    total_samples = sum(node[0] for node in child_nodes)  # Total samples at this depth
    weighted_metric = 0
    
    for num_samples, proportions in child_nodes:
        weight = num_samples / total_samples  # Proportion of samples in this node
        metric = metric_function(proportions)  # Metric of this node
        weighted_metric += weight * metric    # Weighted contribution to overall metric
    
    print(f"Overall {metric_function} is: ", weighted_metric)
    return weighted_metric

def split_data(data, threshold):
    """
    Split the data into two regions based on the threshold.
    Returns x_left, y_left, x_right, y_right
    Example dataset:
    .
    data = np.array([
        [0.2, 2.1], [0.7, 1.5], [1.8, 5.8], [2.2, 6.1], [3.7, 9.1], 
        [4.1, 9.5], [4.5, 9.8], [5.1, 12.7], [6.3, 13.8], [7.4, 15.9]
    ])
    """
    x = data[:, 0]
    y = data[:, 1]
    left_mask = x <= threshold
    right_mask = x > threshold
    
    x_left, y_left = x[left_mask], y[left_mask]
    x_right, y_right = x[right_mask], y[right_mask]
    
    return x_left, y_left, x_right, y_right

def mse_node(y):
    """
    Calculate the mean squared error (MSE) for a node.

    y comes from data[:, 1], i.e. all the y values in the data set.
    """
    if len(y) == 0:
        return 0
    mse = np.mean((y - np.mean(y)) ** 2)
    print("Node MSE is: ", mse)
    return mse

def mse_depth_1(y_left, y_right):
    """
    Calculate the MSE at depth 1, for other depths need to split further

    y_left and y_right are the second and fourth elements returned
    from split_data(data, threshold)
    """
    print("Individual MSEs at depth 1: ")
    # MSE for each region at depth 1
    mse_left = mse_node(y_left)
    mse_right = mse_node(y_right)

    # Weighted MSE at depth 1
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    mse_depth_1 = (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
    print("MSE at depth 1 is: ", mse_depth_1)
    return mse_depth_1