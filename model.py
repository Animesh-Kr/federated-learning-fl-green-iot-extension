# =============================================================================
# model.py
# CNN for CIFAR-10 classification.
#
# This is the UPGRADE model — uses CIFAR-10 (3-channel, 32×32) instead of
# the group baseline's MNIST (1-channel, 28×28). More challenging dataset
# demonstrates the framework's scalability to real-world image complexity.
#
# Architecture:
#   Conv(3→32)+BN+ReLU+Pool → Conv(32→64)+BN+ReLU+Pool →
#   Conv(64→128)+BN+ReLU+Pool → FC(512→256)+Dropout → FC(256→10)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    CNN for CIFAR-10. Input: [B, 3, 32, 32]  Output: [B, 10] logits.

    Upgraded from the group baseline (MNIST CNN) to handle 3-channel
    CIFAR-10 images with an additional conv block and BatchNorm for
    faster convergence in the federated setting.
    """

    def __init__(self) -> None:
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3)

        # 128 × 4 × 4 = 2048 after 3 pool layers on 32×32 input
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → 32×16×16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → 64×8×8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # → 128×4×4
        x = torch.flatten(x, 1)                          # → 2048
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
