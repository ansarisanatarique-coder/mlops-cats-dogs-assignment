import torch
import torch.nn as nn

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()

        # Convolution layers (same as training)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # THIS IS THE MISSING PART
        self.gap = nn.AdaptiveAvgPool2d((1,1))   # converts 26x26 → 1x1

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)          # IMPORTANT
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
