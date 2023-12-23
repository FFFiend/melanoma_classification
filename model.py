"""
Reference class for the CNN architecture used for the classifier.
"""
import torch
import torch.nn as nn

class MelanomaCNN(nn.Module):
    def __init__(self, width=3, normalize_batch=True):
        """
        Init method for the model.
        """
        super(MelanomaCNN, self).__init__()
        self.width = width
        self.bn = normalize_batch
        self.maxpool_layer = nn.MaxPool2d(2,2)
        # first conv2d layer
        self.conv1 = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.width,
            out_channels=32,
            kernel_size=3
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        # 3 batch norm channels corresponding to output num channels of each conv
        # layer.
        if self.bn:
          self.bn1 = nn.BatchNorm2d(self.width)
          self.bn2 = nn.BatchNorm2d(32)
          self.bn3 = nn.BatchNorm2d(64)
          self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(32768 ,256)
        self.fc2 = nn.Linear(256,2)

        # final output should be batch size * 2

    def forward(self,x):
        x = self.maxpool_layer(torch.relu(self.conv1(x)))

        if self.bn:
            x = self.bn1(x)
        
        x = self.maxpool_layer(torch.relu(self.conv2(x)))

        if self.bn:
            x = self.bn2(x)
        
        x = self.maxpool_layer(torch.relu(self.conv3(x)))

        if self.bn:
            x = self.bn3(x)

        x = self.maxpool_layer(torch.relu(self.conv4(x)))

        if self.bn:
          x = self.bn4(x)

        # flatten the image.
        x = x.view(x.shape[0],-1)

        # might swap relu out for smth else
        x = torch.relu(self.fc1(x))
        return self.fc2(x)