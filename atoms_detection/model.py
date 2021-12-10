import torch
from torch import nn


class BasicCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.adaptive = nn.AdaptiveAvgPool2d((3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(3 * 3 * 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(128, num_classes)

        self._initialize_weights()

    # Defining the forward pass
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.fc3.weight, 0, 0.01)


class HeatCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.adaptive = nn.AdaptiveAvgPool2d((3, 3))

        self.fc_layers = nn.Sequential(
            nn.Linear(3 * 3 * 128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc3 = nn.Linear(64, num_classes)

        self._initialize_weights()

    # Defining the forward pass
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.fc3.weight, 0, 0.01)
