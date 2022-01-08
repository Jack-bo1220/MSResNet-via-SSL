import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(FCN,self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.last =  nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda')
    model = FCN(4,4)
    model = model.to(device)
    x = torch.rand((1,4,512,512))
    #x = torch.tensor(x, dtype = torch.float)
    output = model(x)