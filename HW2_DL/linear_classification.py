import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

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
    # Add a column of ones to X for the intercept
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])

    # Calculate the best fit coefficients using the normal equation
    # theta_best = (X_b.T * X_b)^(-1) * X_b.T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

def softmax_safe(s):
    """
    A safe implementation of the softmax function.
    
    Parameters:
    s (np.array): Input array representing the scores or logits, with each column as a data sample.
    
    Returns:
    np.array: Softmax probabilities for each class, with each column as a data sample.
    """
    shift_s = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(shift_s)
    softmax = exp_s / np.sum(exp_s, axis=0, keepdims=True)
    return softmax

def softmax_classifier(X, weights, bias):
    """
    Performs classification on input data using a softmax-based classifier.

    Parameters:
    X (np.array): Input data, a 2D array where each column represents a sample (data point) and each row represents a feature.
    weights (np.array): Weights of the classifier, a 2D array where each row represents the weights for a class.
    bias (np.array): Bias terms for the classifier, a 1D array where each element is the bias for a class.

    Returns:
    tuple: A tuple containing two elements:
           - Probabilities (np.array): A 2D array where each column represents the probability distribution of a sample across different classes.
           - Predicted Labels (np.array): A 1D array of integers, where each element is the predicted class index for the corresponding sample in X.
    """
    # Compute the scores for each class
    scores = np.dot(weights, X) + bias
    # Apply the softmax function to get probabilities
    probabilities = softmax_safe(scores)

    # Determine the class with the highest probability for each sample
    predicted_labels = np.argmax(probabilities, axis=0)

    return probabilities, predicted_labels


def cross_entropy_loss(X, weights, bias, y):
    """
    Compute the cross-entropy loss for a given input, weights, and labels.

    Parameters:
    X (np.array): Input data, a 2D array where each column represents a sample and each row a feature.
    weights (np.array): Weights for the softmax classifier, a 2D array where each row represents a weight vector for a class.
    bias (np.array): bias for the softmax classifier
    y (np.array): True labels, a 1D array of integers where each value is the class index for the corresponding sample in X.

    Returns:
    float: The cross-entropy loss computed for the given input, weights, and labels.
    np.array: Prediction probabilities (hint: use softmax function)
    """
    scores = np.dot(weights, X) + bias
    probabilities = softmax_safe(scores)
    log_probs = -np.log(probabilities[y, np.arange(y.shape[0])])
    loss = np.mean(log_probs)
    return loss, probabilities


def compute_gradients(X, probabilities, y):
    """
    Compute the gradients of the cross-entropy loss with respect to the weights and bias
    in a softmax-based classifier.

    Parameters:
    X (np.array): Input data, a 2D array where each column represents a sample (data point) 
                  and each row represents a feature.
    probabilities (np.array): Output probabilities from the softmax function, a 2D array 
                              where each column represents the probability distribution of 
                              a sample across different classes.
    y (np.array): True labels, a 1D array of integers where each element is the class index 
                  for the corresponding sample in X.

    Returns:
    tuple: A tuple containing two elements:
           - Gradient with respect to the weights (dL_dW), a 2D array with the same shape as the weight matrix.
           - Gradient with respect to the bias (dL_db), a 2D array with the shape (number of classes, 1).
    """
    y_one_hot = np.zeros_like(probabilities)
    y_one_hot[y, np.arange(y.shape[0])] = 1
    dL_dW = np.dot((probabilities - y_one_hot), X.T) / X.shape[1]
    dL_db = np.mean(probabilities - y_one_hot, axis=1, keepdims=True)
    return dL_dW, dL_db

def update_parameters(weights, biases, dL_dW, dL_db, learning_rate):
    """
    Update the weights and biases of a linear classifier.

    This function applies gradient descent to update the model's parameters,
    using the gradients of the loss with respect to weights and biases.

    Parameters:
    - weights: numpy.ndarray
        The current weights of the model.
    - biases: numpy.ndarray
        The current biases of the model.
    - dL_dW: numpy.ndarray
        The gradient of the loss with respect to the weights.
    - dL_db: numpy.ndarray
        The gradient of the loss with respect to the biases.
    - learning_rate: float
        The learning rate to use for the update step.

    Returns:
    - new_weights: numpy.ndarray
        The updated weights after applying gradient descent.
    - new_biases: numpy.ndarray
        The updated biases after applying gradient descent.
    """
    new_weights = weights - learning_rate * dL_dW
    new_biases = biases - learning_rate * dL_db
    return new_weights, new_biases


def plot_loss_curves(loss_arrays, labels, plot_train=True, plot_val=False):
    """
    Plots multiple loss curves on a single figure.

    Parameters:
    - loss_arrays: List of tuples, where each tuple contains a loss array and its label.
    - labels: List of strings, representing labels for each loss array.

    """
    plt.figure(figsize=(10, 6))

    for loss_array, label in zip(loss_arrays, labels):
      if plot_train:
        plt.plot([loss[0] for loss in loss_array], label=label+'[train]')
      if plot_val:
        plt.plot([loss[1] for loss in loss_array], label=label+'[val]')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Learning Rates')
    plt.legend()
    plt.show()

def linear_fit_loss(X, y, slope, intercept):
    """
    Computes the mean squared error (MSE) loss for a linear fit.

    This function calculates the MSE loss for a given set of predictions made by a linear model 
    characterized by a specified slope and intercept. The MSE is computed as the average of the 
    squared differences between the predicted values and the actual target values.

    Parameters:
    - X (numpy.ndarray): A 2D array of input features. Shape should be (n_samples, 1).
    - y (numpy.ndarray): A 2D array of actual target values. Shape should be (n_samples, 1).
    - slope (float): The slope of the linear model.
    - intercept (float): The intercept of the linear model.

    Returns:
    - float: The mean squared error loss.

    Assumptions:
    - The function assumes that both X and y are 2D arrays with a single column (shape (n_samples, 1)).
    
    Raises:
    - AssertionError: If the input arrays X and y do not match the expected shape.
    """
    assert (X.shape[1] == 1)
    assert (y.shape[1] == 1)
    y_pred = slope * X + intercept
    return np.mean((y_pred - y) ** 2)

def plot_loss_landscape(x_train, y_train, slope_range, intercept_range,
                        optimization_stats=None, only_draw_head_and_tail=True):
    """
    Plots the loss landscape for a linear regression model and optionally superimposes the optimization trajectory.

    This function creates a contour plot of the loss landscape defined by a range of slopes and intercepts for a
    linear regression model. It calculates the mean squared error loss for each combination of slope and intercept
    over the provided data. Additionally, if optimization statistics are provided, it plots the trajectory of an
    optimization algorithm on the loss landscape.

    Parameters:
    - x_train (numpy.ndarray): A 2D array of training input features. Shape should be (n_samples, 1).
    - y_train (numpy.ndarray): A 2D array of training target values. Shape should be (n_samples, 1).
    - slope_range (numpy.ndarray): 1D array of slope values to be used in the loss landscape.
    - intercept_range (numpy.ndarray): 1D array of intercept values to be used in the loss landscape.
    - optimization_stats (tuple of lists, optional): A tuple containing three lists: 
        1. A list of loss arrays for each optimization run.
        2. A list of trajectory arrays (each array containing pairs of slope and intercept values) for each optimization run.
        3. A list of labels for each optimization run.
      If None, no optimization trajectory is plotted.
    - only_draw_head_and_tail (bool, optional): If True, only the starting and ending points of each optimization
      trajectory are plotted. If False, the entire trajectory is plotted. Default is True.

    Returns:
    - matplotlib.pyplot: A matplotlib plot object with the plotted loss landscape and optimization trajectories (if provided).

    The function plots the loss landscape as a contour plot where each point's color represents the loss value
    for the corresponding slope and intercept values. The optimization trajectories, if provided, are superimposed
    on this landscape, showing the path taken by the optimizer in the slope-intercept space.
    """


    # Prepare the mesh grid
    S, I = np.meshgrid(slope_range, intercept_range)

    # Initialize a matrix to store the losses
    grid_losses = np.zeros_like(S)

    # Calculate the loss for each combination of slope and intercept
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            y_pred = S[i, j] * x_train + I[i, j]
            grid_losses[i, j] = linear_fit_loss(X=x_train, y=y_train, 
                                          slope=S[i, j], intercept=I[i, j])

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.contourf(S, I, grid_losses, 20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Slope')
    plt.ylabel('Intercept')
    plt.title('Loss Landscape')
    if optimization_stats is not None:
      loss_list, trajectory_list, labels = optimization_stats
      for loss, trajectory, label in zip(loss_list, trajectory_list, labels):
        if only_draw_head_and_tail:
          plt.plot(trajectory[[0, -1], [0, 0]], trajectory[[0, -1], [1, 1]], marker='x', label=label)
        else:
          plt.plot(trajectory[:, 0], trajectory[:, 1], label=label, marker='x')
      plt.legend() 
    return plt


def shuffle_dataset(x_tensor, y_tensor, seed=1024):
    """
    Shuffles the dataset using a fixed seed for reproducibility.

    Parameters:
    - x_tensor (torch.Tensor): A tensor of input features.
    - y_tensor (torch.Tensor): A tensor of output/target values.
    - seed (int, optional): The seed for random number generation. Default is 1024.

    Returns:
    - (torch.Tensor, torch.Tensor): Tuple of shuffled input and output tensors.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(x_tensor.size(0))
    x_tensor_shuffled = x_tensor[perm]
    y_tensor_shuffled = y_tensor[perm]
    return x_tensor_shuffled, y_tensor_shuffled


def batch_gradient_descent(x, y, x_val, y_val, lr=0.01, epochs=100):
    """
    Performs batch gradient descent to fit a linear model.

    This function uses batch gradient descent to optimize the weights (w) and bias (b) 
    of a linear model. The model predicts y as a linear function of x. It minimizes the 
    mean squared error between the predictions and the actual y values over a specified 
    number of epochs.

    Parameters:
    - x (array-like): Input feature array.
    - y (array-like): Output/target array.
    - x_val (array-like): Input feature array (validation).
    - y_val (array-like): Output/target array (validation).
    - lr (float, optional): Learning rate for gradient descent. Default is 0.01.
    - epochs (int, optional): Number of iterations over the entire dataset. Default is 100.

    Returns:
    - losses (numpy.ndarray): Array of loss values at each epoch.
    - trajectory (numpy.ndarray): Array of parameter values (weights and bias) at each epoch.

    The function tracks and returns the trajectory of the parameters (w, b) and the 
    corresponding loss values throughout the optimization process.
    """
    w, b = torch.zeros(1, 1, requires_grad=True), torch.zeros(1, 1, requires_grad=True)
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    def loss_fn(x_tensor, y_tensor, w, b):
      return torch.mean((x_tensor @ w + b - y_tensor) ** 2)

    with torch.no_grad():
      trajectory = [(w.item(), b.item())]
      losses = [(loss_fn(x_tensor, y_tensor, w, b), loss_fn(x_val_tensor, y_val_tensor, w, b))]

    for _ in range(epochs):
        loss = loss_fn(x_tensor, y_tensor, w, b)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            # Manually zero the gradients after updating weights
            w.grad.zero_()
            b.grad.zero_()
            trajectory.append((w.item(), b.item()))
            losses.append((loss_fn(x_tensor, y_tensor, w, b), loss_fn(x_val_tensor, y_val_tensor, w, b)))
            
    return np.array(losses), np.array(trajectory)


def stochastic_gradient_descent(x, y, x_val, y_val, lr=0.01, epochs=100, n_batches=10):
    """
    Performs mini-batch gradient descent to fit a linear model.

    This function optimizes the weights (w) and bias (b) of a linear model using mini-batch
    gradient descent. It minimizes the mean squared error between the predictions and the
    actual y values over a specified number of epochs and batches.

    Parameters:
    - x (array-like): Input feature array (training).
    - y (array-like): Output/target array (training).
    - x_val (array-like): Input feature array (validation).
    - y_val (array-like): Output/target array (validation).
    - lr (float, optional): Learning rate for gradient descent. Default is 0.01.
    - epochs (int, optional): Number of iterations over the entire dataset. Default is 100.
    - n_batches (int, optional): Number of batches to divide the dataset into. Default is 10.

    Returns:
    - losses (numpy.ndarray): Array of loss values at each update.
    - trajectory (numpy.ndarray): Array of parameter values (weights and bias) at each epoch.
    """
    w, b = torch.zeros(1, 1, requires_grad=True), torch.zeros(1, 1, requires_grad=True)
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    def loss_fn(x_tensor, y_tensor, w, b):
        return torch.mean((x_tensor @ w + b - y_tensor) ** 2)

    # Calculating batch size
    batch_size = len(x_tensor) // n_batches

    with torch.no_grad():
      trajectory = [(w.item(), b.item())]
      losses = [(loss_fn(x_tensor, y_tensor, w, b), loss_fn(x_val_tensor, y_val_tensor, w, b))]

    for epoch in range(epochs):
        # Shuffling the dataset
        x_tensor_shuffled, y_tensor_shuffled = shuffle_dataset(x_tensor, y_tensor)

        for i in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor_shuffled[i:i+batch_size]
            y_batch = y_tensor_shuffled[i:i+batch_size]

            loss = loss_fn(x_batch, y_batch, w, b)
            loss.backward()

            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad

                # Manually zero the gradients after updating weights
                w.grad.zero_()
                b.grad.zero_()

                trajectory.append((w.item(), b.item()))
                losses.append((loss_fn(x_tensor, y_tensor, w, b), loss_fn(x_val_tensor, y_val_tensor, w, b)))

    return np.array(losses), np.array(trajectory)

def adam_stochastic_gradient_descent(x, y, x_val, y_val, lr=0.001, epochs=100, n_batches=10):
    """
    Performs mini-batch gradient descent to fit a linear model.

    This function optimizes the weights (w) and bias (b) of a linear model using mini-batch
    gradient descent. It minimizes the mean squared error between the predictions and the
    actual y values over a specified number of epochs and batches.

    Parameters:
    - x (array-like): Input feature array (training).
    - y (array-like): Output/target array (training).
    - x_val (array-like): Input feature array (validation).
    - y_val (array-like): Output/target array (validation).
    - lr (float, optional): Learning rate for gradient descent. Default is 0.01.
    - epochs (int, optional): Number of iterations over the entire dataset. Default is 100.
    - n_batches (int, optional): Number of batches to divide the dataset into. Default is 10.

    Returns:
    - losses (numpy.ndarray): Array of loss values at each update.
    - trajectory (numpy.ndarray): Array of parameter values (weights and bias) at each epoch.
    """
    w, b = torch.zeros(1, 1, requires_grad=True), torch.zeros(1, 1, requires_grad=True)
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    def loss_fn(x_tensor, y_tensor, w, b):
        return torch.mean((x_tensor @ w + b - y_tensor) ** 2)

    # Calculating batch size
    batch_size = len(x_tensor) // n_batches

    with torch.no_grad():
      trajectory = [(w.item(), b.item())]
      losses = [(loss_fn(x_tensor, y_tensor, w, b), loss_fn(x_val_tensor, y_val_tensor, w, b))]

    optimizer = torch.optim.Adam([w, b], lr=lr)

    for epoch in range(epochs):
        # Shuffling the dataset
        x_tensor_shuffled, y_tensor_shuffled = shuffle_dataset(x_tensor, y_tensor)

        for i in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor_shuffled[i:i+batch_size]
            y_batch = y_tensor_shuffled[i:i+batch_size]
            optimizer.zero_grad()
            loss = loss_fn(x_batch, y_batch, w, b)
            loss.backward()
            optimizer.step()
            # Logging
            with torch.no_grad():
                trajectory.append((w.item(), b.item()))
                losses.append((loss_fn(x_tensor, y_tensor, w, b), loss_fn(x_val_tensor, y_val_tensor, w, b)))

    return np.array(losses), np.array(trajectory)
