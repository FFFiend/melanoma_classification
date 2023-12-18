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

        # TODO: veryyyy basic architecture, incomplete. Def. not enough atm,
        # come back to this after exam.

        self.conv1 = nn.Convd2d(
            in_channels=3,
            out_channels=self.num_channels,
            kernel_size=3,
            padding=4
        )

        if normalize_batch:
            self.bn_1 = nn.BatchNorm2d(self.num_channels)


        self.pooling_dim = None
        self.pooling_layer = nn.MaxPool2d(self.pooling_dim, self.pooling_dim)
        

    def forward(self, x):
        pass