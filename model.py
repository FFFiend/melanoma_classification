"""
IMPORTANT! Find site deployed at: 
Reference class for the CNN architecture used for the classifier.
"""
import torch
import torch.nn as nn

class MelanomaCNN(nn.Module):
    def __init__(self, width=3):
        """
        Init method for the model.
        """
        super(MelanomaCNN, self).__init__()
        self.width = width
        self.maxpool_layer = nn.MaxPool2d(2,2)
        #self.maxpool_layer.cuda()
        # first conv2d layer
        self.conv1 = nn.Conv2d(
            in_channels=self.width,
            out_channels=3,
            kernel_size=3,
        )
        #self.conv1.cuda()

        self.temp = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3
        )
        #self.temp.cuda()

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=3
        )
        #self.conv2.cuda()

        self.conv3 = nn.Conv2d(
            in_channels=12,
            out_channels=12,
            kernel_size=3
        )
        #self.conv3.cuda()

        self.conv4 = nn.Conv2d(
            in_channels=12,
            out_channels=24,
            kernel_size=3
        )
        #self.conv4.cuda()

        self.conv5 = nn.Conv2d(
            in_channels=24,
            out_channels=36,
            kernel_size=3
        )
        #self.conv5.cuda()

        self.fc1 = nn.Linear(144,100)
        #self.fc1.cuda()
        self.fc2 = nn.Linear(100,2)
        #self.fc2.cuda()

        # final output should be batch size * 2

    def forward(self,x):
        x = torch.nn.functional.normalize(x,p=3)
        x = self.maxpool_layer(torch.relu(self.conv1(x)))
        #x.cuda()

        x = self.maxpool_layer(torch.relu(self.temp(x)))
        #x.cuda()

        x = self.maxpool_layer(torch.relu(self.conv2(x)))
        #x.cuda()

        x = self.maxpool_layer(torch.relu(self.conv3(x)))
        #x.cuda()

        x = self.maxpool_layer(torch.relu(self.conv4(x)))
        #x.cuda()

        x = self.maxpool_layer(torch.relu(self.conv5(x)))
        #x.cuda()
        # flatten the image.
        x = x.view(x.shape[0],-1)
        #x.cuda()

        # might swap relu out for smth else
        x = torch.relu(self.fc1(x))
        #x.cuda()
        return torch.sigmoid(self.fc2(x))