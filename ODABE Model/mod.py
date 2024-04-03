import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        self.maxp5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1_spatial = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv2_spatial = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv3_spatial = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

        self.linear1 = nn.Linear(in_features=12 * 12 * 256, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=2)

        self._initialize_weights()