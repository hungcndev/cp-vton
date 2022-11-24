from inin_weights import init_weights

import torch
import torch.nn as nn

from typing import Optional, List

# -------------------------------------------------------------------------- #

class FeatureExtraction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[List[int]] = None,
        normalize: nn = nn.BatchNorm2d
       ):
        super().__init__()

        # H) 256 -> 128 -> 64 -> 32 -> 16 -> 16 -> 16
        # W) 192 ->  96 -> 48 -> 24 -> 12 -> 12 -> 12
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]

        layers = []

        for h_channel in hidden_channels:
            layers += [
                nn.Conv2d(in_channels, h_channel, 4, 2, 1, bias=False),
                normalize(h_channel),
                nn.ReLU(True),
            ]

            if h_channel == hidden_channels[-1]:
                layers += [
                    nn.Conv2d(h_channel, h_channel, 3, 1, 1, bias=False),
                    normalize(h_channel),
                    nn.ReLU(True),

                    nn.Conv2d(h_channel, h_channel, 3, 1, 1, bias=False),
                    normalize(h_channel),
                    nn.ReLU(True)
                ]
            else:
                in_channels = h_channel

        self.model = nn.Sequential(*layers)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)

class FeatureL1Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature):
        # L1norm
        # out = sum(x_i)

        epsilon = 1e-6
        l1norm = (torch.sum(feature, 1) + epsilon).unsqueeze(1).expand_as(feature)
        return torch.div(feature, l1norm)

class FeatureL2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature):
        # L2norm
        # out = (sum((x_i)^2))^1/2
        # out = torch.pow(sum((x_i)^2)), 0.5)
        # out = torch.pow(sum(torch.pow(x, 2)), 0.5)
        # out = torch.pow(torch.sum(torch.pow(x, 2), 1), 0.5)
        epsilon = 1e-6
        l2norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, l2norm)

class FeatureCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, featureA, featureB):
        b, c, h, w = featureA.size()
        
if __name__ == "__main__":
    feature = torch.randn(4, 512, 16, 12)
    print(feature)
    x = torch.sum(torch.pow(feature, 2), 1) + 1e-06
    print(x.shape)
    exp = 0.5 # 1/2
    out = torch.pow(x, exp).unsqueeze(1).expand_as(feature)
    out = torch.div(feature, out)
    print(out.shape)