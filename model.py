"""
Reference class for the CNN architecture used for the classifier.
"""
import torch
import torch.nn as nn

class MelanomaCNN(nn.Module):
    def __init__(self, num_channels, normalize_batch=True):
        """
        Init method for the model.
        """
        super(MelanomaCNN, self).__init__()

        self.num_channels = num_channels
        self.normalize_batch = normalize_batch

        # TODO: incomplete architecture, 
        # FINAL DELIVERABLE complete on Dec 24th post exams.

        self.conv1 = nn.Convd2d(
            in_channels=3,
            out_channels=self.num_channels,
            kernel_size=9,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.num_channels * 2,
            out_channels = None,
            kernel_size = 9
        )

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3
        )

        self.conv3 = nn.Conv2d(
            in_channels = 256
        )

        if normalize_batch:
            self.bn_1 = nn.BatchNorm2d(self.num_channels)

        self.dropout_layer = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

        

    def forward(self, x):
        # TODO x.resize()
        x = self.maxpool1(torch.relu(self.conv1(x)))
        if self.bn_1:
            x = self.bn_1(x)
        x = self.maxpool1(torch.relu(self.conv1(x)))
        if self.bn_1:
            x = self.bn_1(x)
        x = self.maxpool1(torch.relu(self.conv2(x)))

        x = self.dropout_layer(x)

        if self.bn_1:
            x = self.bn_1(x)
        x = self.maxpool1(torch.relu(self.conv3(x)))
        if self.bn_1:
            x = self.bn_1(x)
        
        x = x.view(-1, self.num_channels)
        x = torch.relu(self.fc1(x))
        return x