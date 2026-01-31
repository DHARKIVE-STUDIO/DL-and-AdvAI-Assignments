import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import random

class TwoLayerNet(nn.Module):
    """
    A simple two-layer neural network with ReLU and Softmax activation.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Second fully connected layer.
        softmax (nn.Softmax): Softmax activation function.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The number of units in the hidden layer.
        output_size (int): The size of the output features.
    """
    def __init__(self, input_size, hidden_size, output_size):
      """
      Initializes the TwoLayerNet model with specified layer sizes.
      """
      super(TwoLayerNet, self).__init__()
      random.seed(0)
      torch.manual_seed(0)
      # Defining the layers
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_size, output_size)
      self.softmax = nn.Softmax(dim=1)
      
      # Correct weight initialization
      init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
      init.constant_(self.fc1.bias, 0)
      init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
      init.constant_(self.fc2.bias, 0)

    def forward(self, x):
      """
      Defines the forward pass of the TwoLayerNet.

      Args:
          x (Tensor): Input tensor.

      Returns:
          Tensor: The output of the network.
      """
      out = self.fc1(x)
      out = self.relu(out)  # Apply ReLU activation
      out = self.fc2(out)
      return self.softmax(out)  # Softmax should be applied only at the output

    def calculate_loss(self, output, labels):
      """
      Calculates the loss between the model's output and the true labels.

      This method uses the Cross Entropy Loss, which is commonly used for classification tasks.
      
      Args:
          output (Tensor): The logits output by the model. It's the raw, unnormalized scores for each class.
          labels (Tensor): The true labels for the input data. These should be indices of the correct classes.

      Returns:
          Tensor: The loss value as a PyTorch tensor.
      """
      # Define the criterion for loss calculation - Cross Entropy Loss
      criterion = nn.CrossEntropyLoss()
      return criterion(output, labels)

    def backward(self, output, labels, retain_graph=False):
      """
      Calculates the loss and Performs the backward pas

      This method first computes the loss using the `calculate_loss` method. Then, it
      zeroes out the existing gradients to prevent accumulation. Finally, it performs 
      the backward pass to compute the gradients.

      Args:
          output (Tensor): The output tensor from the model's forward pass.
          labels (Tensor): The true labels for the input data.

      Returns:
          Tensor: The computed loss value.
      """
      loss = self.calculate_loss(output, labels)
      self.zero_grad()  # Reset gradients
      loss.backward(retain_graph=retain_graph)
      return loss

    def update_params(self, learning_rate):
      """
      Updates the parameters of the model using Gradient Descent.

      This method applies a simple update rule to the weights and biases of the
      layers. It subtracts the product of the learning rate and the gradient from
      each parameter.

      Args:
          learning_rate (float): The learning rate for the parameter update.
      """

      with torch.no_grad():
        for param in self.parameters():
            param -= learning_rate * param.grad
      self.zero_grad()  # Reset gradients after updating

    def train(self, X, labels, learning_rate, number_of_training_iterations, verbose=False):
      """Trains the model using the provided data.

      The training process involves multiple iterations of forward and backward passes
      and parameter updates. The initial and final loss values are printed for reference.

      Args:
          X (Tensor): The input data.
          labels (Tensor): The true labels corresponding to the input data.
          learning_rate (float): The learning rate for parameter updates.
          number_of_training_iterations (int): The number of iterations to train the model.

      Returns:
          Tensor: The loss value after the final training iteration.
      """
      output = self.forward(X)
      loss = self.calculate_loss(output, labels)
      print(f"Initial Loss: {loss.item()}")

      for i in range(number_of_training_iterations):  # Use correct variable name
        output = self.forward(X)
        loss = self.backward(output, labels)
        self.update_params(learning_rate)
        if verbose:
            print(f"Iteration {i}, Loss: {loss.item()}")

      print(f"Final Loss: {loss.item()}")
      return loss
