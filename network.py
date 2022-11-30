import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image

import time


class MultiClassFF(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MultiClassFF, self).__init__()
        '''
        Hidden size = list[]
        '''
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_class = num_class
        self.layer_size = len(self.hidden_size)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_size[0])])
        self.linears.extend([torch.nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(self.layer_size-1)])
        self.linears.append(torch.nn.Linear(hidden_size[self.layer_size-1], self.num_class))

        self.relu = torch.nn.ReLU()
        #self.sorfmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        for i in range(self.layer_size+1):
            x = self.linears[i](x)
            if i != self.layer_size:
                x = self.relu(x)
        #x = self.sorfmax(x)
        return x

class dilated_CNN_61(torch.nn.Module):
    def __init__(self, num_class, inp):
        super(dilated_CNN_61, self).__init__()
        self.conv1 = torch.nn.Conv2d(inp, 128, 1, stride = 1, padding = 0, dilation=1)
        self.conv2 = torch.nn.Conv2d(128, 64, 3, stride = 1, padding = 2, dilation=2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 4, dilation=4)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 8, dilation=8)
        self.conv5 = torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 1, dilation=1)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 2, dilation=2)
        self.conv7 = torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 4, dilation=4)
        self.conv8 = torch.nn.Conv2d(64, 32, 3, stride = 1, padding = 8, dilation=8)
        self.conv9 = torch.nn.Conv2d(32, num_class, 3, stride = 1, padding = 1, dilation=1)

        # self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = self.conv9(x)
        # y = self.softmax(x) #Log
        return x
