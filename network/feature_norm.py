import torch
import torch.nn as nn

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

def feature_norm(norm_type: str):
    if norm_type == "L1":
        return FeatureL1Norm()
    elif norm_type == "L2":
        return FeatureL2Norm()
    else:
        raise Exception('Invalid string entered. Enter L1 or L2')