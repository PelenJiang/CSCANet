from torch import nn
import torch
import math
import torch.nn.functional as F
from math import log

class ChannelAtt(nn.Module):
    def __init__(self, channel,gamma=2, b=1):
        super(ChannelAtt, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(channel, 2) + self.b)/ self.gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        scale1 = x1 + x2
        scale1 = self.conv(scale1.squeeze(-1).transpose(-1,-2))
        scale1 = scale1.transpose(-1,-2).unsqueeze(-1)
        scale1 = self.sigmoid(scale1)
        return scale1

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialAtt(nn.Module):
    def __init__(self):
        super(SpatialAtt, self).__init__()
        self.compress = ChannelPool()
        self.spatial= nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale2 = torch.sigmoid(x_out) # broadcasting
        return scale2

class CSCAModule(nn.Module):
    def __init__(self, channels):
        super(CSCAModule, self).__init__()
        self.ChannelAtt = ChannelAtt(channels)
        self.SpatialAtt = SpatialAtt()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_scale1 = self.ChannelAtt(x)
        x_scale2 = self.SpatialAtt(x)
        x_out = x * x_scale1.expand_as(x) * x_scale2.expand_as(x)

        return x_out

        # if __name__ == '__main__':
#     input=torch.randn(2,64,7,7)
#     CSCAM = CSCAModule(channels=64)
#     output=CSCAM(input)
#     print(output.shape)