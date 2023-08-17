# stdlib
from typing import List, Type

# third party
import torch
import torch.nn as nn
import torch.nn.functional as F


# This is a neural network class with two dropout layers used for image classification.
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # add two dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, embed: bool = False) -> torch.Tensor:
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): The input tensor
            embed (bool, optional): Flag to identify if the output is an embedding or not. Defaults to False.

        Returns:
            torch.Tensor: The output tensor
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        embedding = torch.relu(self.fc2(x))
        embedding = self.dropout2(embedding)
        logits = self.fc3(embedding)

        if embed:
            return embedding
        else:
            return logits


# The LeNet class is a neural network model
class LeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, embed: bool = False) -> torch.Tensor:
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): The input tensor
            embed (bool, optional): Flag to identify if the output is an embedding or not. Defaults to False.

        Returns:
            torch.Tensor: The output tensor
        """
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        logits = self.fc3(embedding)

        if embed:
            return embedding
        else:
            return logits


# The LeNetMNIST class is a neural network model for use on the MNIST dataset.
class LeNetMNIST(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, embed: bool = False) -> torch.Tensor:
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): The input tensor
            embed (bool, optional): Flag to identify if the output is an embedding or not. Defaults to False.

        Returns:
            torch.Tensor: The output tensor
        """
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        embedding = F.relu(self.fc1(x))
        logits = self.fc2(embedding)

        if embed:
            return embedding
        else:
            return logits


"""ResNet in PyTorch.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""


class Block(nn.Module):
    expansion: int = 1


class BasicBlock(Block):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicBlock neural network

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(Block):
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicBlock neural network

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# This is a ResNet model for MNIST dataset
class ResNetMNIST(nn.Module):
    def __init__(
        self, block: Type[Block], num_blocks: List[int], num_classes: int = 10
    ) -> None:
        super(ResNetMNIST, self).__init__()
        self.in_planes: int = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # add two dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)

    def _make_layer(
        self, block: Type[Block], planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        """
        Make a layer of the ResNet model

        Args:
            block (nn.Module): The block to be used to make the layer of the ResNet model
            planes (int): The number of planes
            num_blocks (int): The number of blocks
            stride (int): The stride

        Returns:
            nn.Module: The layer of the ResNet model
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, embed: bool = False) -> torch.Tensor:
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): The input tensor
            embed (bool, optional): Flag to identify if the output is an embedding or not. Defaults to False.

        Returns:
            torch.Tensor: The output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.dropout3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedding = out.view(out.size(0), -1)
        logits = self.linear(embedding)
        if embed:
            return embedding
        else:
            return logits


# This is a ResNet model
class ResNet(nn.Module):
    def __init__(
        self, block: Type[Block], num_blocks: List[int], num_classes: int = 10
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes: int = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # add two dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)

    def _make_layer(
        self, block: Type[Block], planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        """
        Make a layer of the ResNet model

        Args:
            block (nn.Module): The block to be used to make the layer of the ResNet model
            planes (int): The number of planes
            num_blocks (int): The number of blocks
            stride (int): The stride

        Returns:
            nn.Module: The layer of the ResNet model
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, embed: bool = False) -> torch.Tensor:
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): The input tensor
            embed (bool, optional): Flag to identify if the output is an embedding or not. Defaults to False.

        Returns:
            torch.Tensor: The output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.dropout3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedding = out.view(out.size(0), -1)
        logits = self.linear(embedding)
        if embed:
            return embedding
        else:
            return logits


def ResNet18MNIST() -> nn.Module:
    """
    Returns:
        nn.Module: ResNet18 model for MNIST dataset
    """
    return ResNetMNIST(BasicBlock, [2, 2, 2, 2])


def ResNet18() -> nn.Module:
    """
    Returns:
        nn.Module: ResNet18 model for CIFAR10 dataset
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])
