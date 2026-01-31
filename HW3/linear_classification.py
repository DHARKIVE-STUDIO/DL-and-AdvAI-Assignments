import numpy as np

def flatten_array(array):
    """
    Flattens a array into a 1D vector.

    Parameters:
    array (np.array): A NumPy array of shape (x, y, z).

    Returns:
    np.array: A 1D vector containing all elements of the original array.
    """
    return array.flatten()

def logistic_function(z):
    """Apply the logistic function.

    Parameters:
    z (float): The linear combination of inputs and coefficients.

    Returns:
    float: The probability after applying the logistic function.
    """
    return 1 / (1 + np.exp(-z))

def logistic_classification(input_vector, coefficients):
    """Perform logistic classification on an input vector using given coefficients.

    The first element of the coefficients array is considered as the bias term.

    Parameters:
    input_vector (np.array): An n-dimensional vector of input features.
    coefficients (np.array): An (n+1)-dimensional vector where the first element is the bias,
                             and the rest are the feature coefficients.

    Returns:
    tuple: The probability.
    """
    z = np.dot(input_vector, coefficients[1:]) + coefficients[0]  # Bias + Weighted sum
    probability = logistic_function(z)
    return probability


def softmax(x):
    """
    Compute the softmax of vector x.
    

    Parameters:
    x (np.array): A 1D numpy array.

    Returns:
    np.array: Softmax of the input array.
    """
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

  
def safe_softmax(x):
    """
    Compute the softmax of vector x in a numerically safe manner by subtracting the max value
    before exponentiation to avoid potential overflow issues.

    Parameters:
    x (np.array): A 1D numpy array.

    Returns:
    np.array: Softmax of the input array.
    """
    shift_x = x - np.max(x)
    exp_shift_x = np.exp(shift_x)
    return exp_shift_x / np.sum(exp_shift_x)

def softmax_classifier(input_vector, weights):
    """
    Classify an input vector using a softmax-based classifier.

    The weights matrix should have the same number of columns as the length of the input vector.
    The first column of the weights matrix is treated as the bias for each class.

    Parameters:
    input_vector (np.array): A 1D numpy array representing the input features.
    weights (np.array): A 2D numpy array where the first column represents bias terms
                        and the remaining columns represent the weights for each class.

    Returns:
    int: The predicted class index.
    np.array: The probabilities for each class.
    """
    scores = np.dot(weights[:, 1:], input_vector) + weights[:, 0]  # Bias + Weighted sum
    probabilities = softmax(scores)
    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities
