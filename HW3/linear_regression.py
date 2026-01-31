import numpy as np

def linear_predict_1d(x, slope, intercept):
    """
    Predicts the value of a linear function given an input, slope, and intercept.

    Args:
    x (float): The input value for which the prediction is to be made.
    slope (float): The slope of the linear function.
    intercept (float): The y-intercept of the linear function.

    Returns:
    float: The predicted value of the linear function for the given input.
    """
    return slope * x + intercept

def linear_fit_1d(X, y):
    """
    Perform a linear fit for one-dimensional input.

    Parameters:
    X (np.array): A 1D array representing the independent variable.
    y (np.array): A 1D array of the same length as X, representing the target variable.

    Returns:
    np.array: Coefficients of the linear model [intercept, slope].
    """
    X_mean, y_mean = np.mean(X), np.mean(y)
    slope = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    intercept = y_mean - slope * X_mean
    return np.array([intercept, slope])


def linear_predict_nd(data_point, parameters):
    """
    Predict the output for a single n-dimensional data point using a linear model.
    
    :param data_point: A NumPy array representing the single n-dimensional data point.
    :param parameters: A NumPy array representing the parameters of the model (intercept and coefficients).
                       The first element is assumed to be the intercept.
    :return: The prediction as a float.
    """
    return np.dot(data_point, parameters[1:]) + parameters[0]  # Bias + Weighted sum

def linear_fit_nd(X, y):
    """
    Perform a linear fit for n-dimensional input.

    Parameters:
    X (np.array): A 2D array of shape (m, n) where m is the number of observations
                  and n is the number of features/dimensions.
    y (np.array): A 1D array of length m, representing the target variable.

    Returns:
    np.array: Coefficients of the linear model.
    """
    X_aug = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    theta_best = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y  # Normal equation
    return theta_best
