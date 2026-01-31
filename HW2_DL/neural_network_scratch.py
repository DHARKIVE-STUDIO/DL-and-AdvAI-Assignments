import numpy as np
import matplotlib.pyplot as plt
import torch
import random

class TwoLayerNet(object):
  def __init__(self, input_size, hidden_size, output_size,
               dtype=torch.float64, device='cpu', std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (H, D)
    b1: First layer biases; has shape (H, 1)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C, 1)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    - dtype: Optional, data type of each initial weight params
    - device: Optional, whether the weight params is on GPU or CPU
    - std: Optional, initial weight scaler.
    """
    # reset seed before start
    random.seed(0)
    torch.manual_seed(0)
    self.params = {
      'W1': std * torch.randn(hidden_size, input_size, dtype=dtype, device=device),
      'b1': torch.zeros((hidden_size, 1), dtype=dtype, device=device),
      'W2': std * torch.randn(output_size, hidden_size, dtype=dtype, device=device),
      'b2': torch.zeros((output_size, 1), dtype=dtype, device=device),
    }

  def relu(self, X):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    This function computes the ReLU of the input tensor `X` by replacing
    every negative value in `X` with zero, effectively clipping at zero.

    Parameters:
    - X (torch.Tensor): The input tensor on which the ReLU function will be applied.

    Returns:
    - torch.Tensor: A new tensor with the ReLU function applied to each element of `X`.
    """
    return torch.maximum(torch.tensor(0, dtype=X.dtype, device=X.device), X)

  def softmax(self, x):
    """
    Applies the softmax function to the input tensor `x`.

    The softmax function is applied to each column of the input tensor, converting
    the input into a distribution of probabilities over different classes. To improve
    numerical stability, the maximum value of each column is subtracted from every element
    within that column before applying the exponential function.

    Parameters:
    - x (torch.Tensor): The input tensor. The softmax function will be applied across the 0th dimension,
      treating each column as a separate set of logits.

    Returns:
    - torch.Tensor: The resulting tensor after applying the softmax function to `x`. Each column of the
      output tensor represents a probability distribution over the classes for the corresponding column
      of the input tensor.
    """
    shift_x = x - torch.max(x, dim=0, keepdim=True)[0]
    exp_x = torch.exp(shift_x)
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

  def generate_one_hot(self, shape, labels):
    """
    Generates a one-hot encoded tensor based on the specified shape and labels.

    This function creates a tensor of zeros with the given shape, and then sets 
    ones in positions indicated by the `labels` tensor. The `labels` tensor contains
    indices at which the one-hot encoding will place ones. This is particularly useful
    for converting class labels into a format suitable for classification tasks.

    Parameters:
    - shape (tuple): The shape of the one-hot encoded tensor to be generated. The number of rows
      should correspond to the number of classes, and the columns
      should match the number of samples.
    - labels (torch.Tensor): A tensor containing the indices where ones should be placed.
      Its size should match the columns of the `shape` parameter.

    Returns:
    - torch.Tensor: A one-hot encoded tensor with ones placed according to the `labels`
      indices and zeros elsewhere.
    """
    one_hot = torch.zeros(shape, dtype=torch.float64)
    one_hot[labels, torch.arange(labels.shape[0])] = 1
    return one_hot

  def cross_entropy_loss(self, probabilities, labels):
    """
    Compute the cross-entropy loss from scratch.
    
    Parameters:
    - probabilities: A tensor of shape (N, M) containing probabilities, where N is the number of classes
      and M is the number of samples.
    - labels: A tensor of shape (M,) containing the indices of the true class for each sample.
    
    Returns:
    - loss: The computed cross-entropy loss.
    """
    log_probs = -torch.log(probabilities[labels, torch.arange(labels.shape[0])])
    return torch.mean(log_probs)

  def forward(self, X):
    """
    Performs the forward pass of the neural network.

    This method computes the forward propagation of the input tensor `X` through two linear layers 
    of the neural network, applying a ReLU activation function after the first linear transformation 
    and a softmax function after the second linear transformation to produce a probability distribution 
    over the output classes.

    The process involves matrix multiplication of the input with the weight matrices (`W1` and `W2`), 
    addition of bias terms (`b1` and `b2`), application of the ReLU activation function after the first 
    layer, and application of the softmax function after the second layer to obtain the final output 
    probabilities.

    Parameters:
    - X (torch.Tensor): The input tensor to the neural network. The shape of `X` should be compatible 
      with the weight matrix of the first layer (`W1`).

    Returns:
    - torch.Tensor: The output tensor containing the predicted probabilities for each class. The output 
      is obtained after applying the softmax function to the activations of the second layer.
    """
    self.s1 = self.params['W1'] @ X + self.params['b1']
    self.h1 = self.relu(self.s1)
    self.s2 = self.params['W2'] @ self.h1 + self.params['b2']
    self.p = self.softmax(self.s2)
    return self.p
    

  def forward_backward(self, X, labels):
    """
    Performs the forward and backward passes of the neural network.

    This method first conducts a forward pass using the input tensor `X`, obtaining
    the predicted probabilities for each class. It then calculates the cross-entropy loss
    using the predicted probabilities and the true class labels. Following the loss calculation,
    the method performs a backward pass to compute the gradients of the loss with respect to
    the network's parameters (weights and biases).

    During the backward pass, gradients are computed using the chain rule. A one-hot encoding
    of the labels is generated to facilitate the gradient calculation for the cross-entropy loss.
    The method updates the gradients of the weights and biases stored in the network's parameters.

    Parameters:
    - X (torch.Tensor): The input tensor to the neural network, where the second dimension (X.shape[1])
      represents the number of samples.
    - labels (torch.Tensor): A tensor containing the indices of the true class labels for each sample.

    Returns:
    - float: The cross-entropy loss computed from the predicted probabilities and true labels.

    Note:
    The computed gradients are stored as attributes of the class and can be accessed after calling
    this method for updating the network's parameters during the optimization process.
    """
    ##############################################################################
    #                    TODO: Write you code                                    #
    ##############################################################################
    probabilities = self.forward(X)
    loss = self.cross_entropy_loss(probabilities, labels)

    # Compute gradients
    y_one_hot = self.generate_one_hot(probabilities.shape, labels)
    dL_ds2 = probabilities - y_one_hot
    self.dL_dW2 = dL_ds2 @ self.h1.T / X.shape[1]
    self.dL_db2 = torch.mean(dL_ds2, dim=1, keepdim=True)

    dL_dh1 = self.params['W2'].T @ dL_ds2
    dL_ds1 = dL_dh1 * (self.s1 > 0).float()
    self.dL_dW1 = dL_ds1 @ X.T / X.shape[1]
    self.dL_db1 = torch.mean(dL_ds1, dim=1, keepdim=True)

    return loss

  def training(self, X, labels, learning_rate, iterations):
    losses = []
    for i in range(iterations):
      loss = self.forward_backward(X, labels)
      losses.append(loss)
      self.params['W2'] -= learning_rate * self.dL_dW2
      self.params['b2'] -= learning_rate * self.dL_db2
      self.params['b1'] -= learning_rate * self.dL_db1
      self.params['W1'] -= learning_rate * self.dL_dW1
    return losses
  
  def numerical_gradient(self, X, labels):
    small_number = 1e-6
    grads = {}
    for param_name in ['b2', 'W2', 'b1', 'W1']:
        param = self.params[param_name]
        grads[param_name] = torch.zeros_like(param)
        grad = grads[param_name]
        # It's important to clone the parameter for a correct "copy" to reset it later
        param_original = param.clone()
        nrows, ncols = param.shape
        for row in range(nrows):
            for col in range(ncols):
                # Save the original value
                original_value = param[row, col].item()

                # Perturb parameter
                param[row, col] = original_value + small_number
                loss_after = self.cross_entropy_loss(self.forward(X), labels)

                # Perturb parameter
                param[row, col] = original_value - small_number
                loss_before = self.cross_entropy_loss(self.forward(X), labels)

                # Compute gradient
                grad[row, col] = (loss_after - loss_before) / (2 * small_number)

                # Reset the parameter to its original value
                param[row, col] = original_value
    return grads

  def save(self, path):
    torch.save(self.params, path)
    print("Saved in {}".format(path))

  def load(self, path):
    self.params = torch.load(path, map_location='cpu')
    print("load checkpoint file: {}".format(path))