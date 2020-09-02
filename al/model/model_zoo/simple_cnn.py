import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self, classes=10):
        super(ConvModel, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ])
        self.fc = nn.Sequential(*[
            nn.Dropout(0.5),
            nn.Linear(7*7*16, classes)
        ])

    def forward(self, x, features=False):
        feature = self.conv(x)
        if features:
            return self.fc(feature), feature
        return self.fc(feature)
