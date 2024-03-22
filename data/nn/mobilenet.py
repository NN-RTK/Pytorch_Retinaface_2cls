# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(int(inp), int(oup), 3, stride, 1, bias=False),
                nn.BatchNorm2d(int(oup)),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                #depthwise sep
                nn.Conv2d(int(inp), int(inp), 3, stride, 1, groups=int(inp), bias=False),
                nn.BatchNorm2d(int(inp)),
                nn.ReLU(inplace=True),

                nn.Conv2d(int(inp), int(oup), 1, 1, 0, bias=False),
                nn.BatchNorm2d(int(oup)), 
                nn.ReLU(inplace=True),
            )
        alpha = 0.25
        self.model = nn.Sequential(
            conv_bn(3, 32*alpha, 2),
            conv_dw(32*alpha, 64*alpha, 1),
            conv_dw(64*alpha, 128*alpha, 2),
            conv_dw(128*alpha, 128*alpha, 1),
            conv_dw(128*alpha, 256*alpha, 2),
            conv_dw(256*alpha, 256*alpha, 1),
            
            conv_dw(256*alpha, 512*alpha, 2),
            conv_dw(512*alpha, 512*alpha, 1),
            conv_dw(512*alpha, 512*alpha, 1),
            conv_dw(512*alpha, 512*alpha, 1),
            conv_dw(512*alpha, 512*alpha, 1),
            conv_dw(512*alpha, 512*alpha, 1),
            
            conv_dw(512*alpha, 1024*alpha, 2),
            conv_dw(1024*alpha, 1024*alpha, 1),
        )
        self.fc = nn.Linear(int(1024*alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x