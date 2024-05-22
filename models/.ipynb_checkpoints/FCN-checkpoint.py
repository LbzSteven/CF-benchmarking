import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_shape, out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.global_avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class FCN_AE(nn.Module):
    def __init__(self, input_shape, input_size):
        super(FCN_AE, self).__init__()

        self.ec1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, out_channels=16, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, padding='same'),
            nn.ReLU(),
        )
        self.ec2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
        )

        self.ec3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.dc1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(size=int(input_size/4)),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.dc2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Upsample(size=int(input_size/2)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding='same'),
            nn.ReLU(),
        )

        self.dc3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Upsample(size=input_size),
            nn.Conv1d(in_channels=16, out_channels=input_shape, kernel_size=8, padding='same'),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.ec3(self.ec2(self.ec1(x)))
        x = self.dc3(self.dc2(self.dc1(x)))
        return x
