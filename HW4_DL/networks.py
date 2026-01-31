import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Define LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        set_seed(1024)
        super(LeNet5, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)  # 1 input channel (grayscale)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        # Define fully connected layers
        self.fc1 = nn.Linear(50 * 4 * 4, 500)  # MNIST image size reduced to 4x4
        self.fc2 = nn.Linear(500, 10)  # 10 output classes

        # Initialize weights
        init.kaiming_normal_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0)
        init.kaiming_normal_(self.conv2.weight)
        init.constant_(self.conv2.bias, 0)
        init.kaiming_normal_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        init.kaiming_normal_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 50 * 4 * 4)  # Flatten for FC layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the BasicBlock for ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet-18 class
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # Assuming MNIST input is 1 channel
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

import torch

def cal_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of the model on the given data loader.

    Args:
        model: The PyTorch model.
        data_loader: The PyTorch data loader.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        The accuracy of the model.
    """
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy