from inin_weights import init_weights
from feature_norm import feature_norm

import torch
import torch.nn as nn

import numpy as np

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

class FeatureCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    # batch x 512 x 16 x 12 -> batch x 192 x 16 x 12
    def forward(self, featureA, featureB):
        b, c, h, w = featureA.size()

        # featureA -> batch x 512 x 16 x 12 -> batch x 512 x 192
        # featureB -> batch x 512 x 16 x 12 -> batch x 192 x 512
        featureA = featureA.transpose(2, 3).contiguous().view(b, c, h*w)
        featureB = featureB.view(b, c, h*w).transpose(1, 2)

        feature_mul = torch.bmm(featureB, featureA) # batch x 192 x 192
        # batch x 192 x 192 -> batch x 16 x 12 x 192 -> batch x 16 x 192 x 12 -> batch x 192 x 16 x 12
        correlation_tensor = feature_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor # 4x192x16x12

class FeatureRegression(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: Optional[List] = None,
        hidden_features: Optional[List] = None,
        out_feature: int = 6
        ):
        super().__init__()

        def conv(in_channels, hidden_channels: Optional[List] = None):
            if hidden_channels is None:
                hidden_channels = [512, 256, 128, 64]

            conv_layers = []
            k, s, p = 4, 2, 1

            for h_channel in hidden_channels:
                conv_layers += [
                    nn.Conv2d(in_channels, h_channel, k, s, p, bias=False),
                    nn.BatchNorm2d(h_channel),
                    nn.ReLU(True)
                ]
                if h_channel == hidden_channels[1]:
                    k, s, p = 3, 1, 1

                in_channels = h_channel

            return nn.Sequential(*conv_layers)

        def linear(in_feature: int = 64 * 4 * 3, hidden_features: Optional[List] = None, out_feature:int = 50):
            if hidden_features is None:
                hidden_features = [out_feature]

            linear_layers = []

            for h_feature in hidden_features:
                linear_layers += [
                    nn.Linear(in_feature, out_feature, bias=False),
                    ]
                if h_feature == hidden_features[-1]:
                    linear_layers += [
                    nn.Tanh()
                    ]
                else:
                    in_feature = h_feature

            return nn.Sequential(*linear_layers)

        self.conv = conv(in_channels, hidden_channels)
        self.linear = linear(64*4*3, hidden_features, out_feature)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        
        return x

class TpsGridGen(nn.Module):
    def __init__(
        self,
        grid_size: int = 5,
        out_hight: int = 256,
        out_weight: int = 192,
        use_regular_grid: bool = True
        ):
        super().__init__()

        # grid = torch.zeros((3, out_hight, out_weight), dtype=torch.float)
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, steps=out_weight), torch.linspace(-1, 1, steps=out_hight), indexing="ij")
        self.grid_x = grid_x.unsqueeze(0).unsqueeze(-1)
        self.grid_y = grid_y.unsqueeze(0).unsqueeze(-1)

        if use_regular_grid:
            axis_coords = torch.linspace(-1, 1, steps=grid_size)
            self.N = grid_size ** 2

            p_y, p_x = torch.meshgrid(axis_coords, axis_coords, indexing="ij")
            p_x = torch.reshape(p_x, (-1, 1))
            p_y = torch.reshape(p_y, (-1, 1))
            self.p_x_base = p_x.clone()
            self.p_y_base = p_y.clone()
            self.Li = self.compute_L_inverse(p_x, p_y)
            self.p_x = p_x.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.p_y = p_x.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)

    def compute_L_inverse(self, p_x, p_y):
        N = p_x.size(0)
        p_x_mat = p_x.expand(N, N)
        p_y_mat = p_y.expand(N, N)
        p_dist_squared = torch.pow(p_x_mat - p_x_mat.transpose(0, 1), 2) + torch.pow(p_y_mat - p_y_mat.transpose(0, 1), 2)
        p_dist_squared[p_dist_squared == 0] = 1
        K = torch.mul(p_dist_squared, torch.log(p_dist_squared))
        p_x_ones = torch.ones_like(p_x, dtype=torch.float)
        p_y_zeros = torch.zeros_like(p_y, dtype=torch.float)
        P = torch.cat((p_x_ones, p_x, p_y), 1)
        L = torch.cat((torch.cat((K, P), 1)), torch.cat((P.transpose(0, 1), p_y_zeros), 1), 0)
        Li = torch.inverse(L)

        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)

        batch_size = theta.size(0)
        Q_x = theta[:, : self.N, :, :].squeeze(3)
        Q_y = theta[:, self.N :, :, :].squeeze(3)

        Q_x = Q_x + self.p_x_base.expand_as(Q_x)
        Q_y = Q_y + self.p_y_base.expand_as(Q_y)

        b, h, w, _ = points.size()

        P_x = self.p_x.expand((1, h, w, 1, self.N))
        P_y = self.p_y.expand((1, h, w, 1, self.N))

        W_x = torch.bmm(self.Li[:, : self.N, :self.N].expand((batch_size, self.N, self.N)), Q_x)
        W_y = torch.bmm(self.Li[:, : self.N, :self.N].expand((batch_size, self.N, self.N)), Q_y)

        W_x = W_x.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, h, w, 1, 1)
        W_y = W_y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, h, w, 1, 1)

        A_x = torch.bmm(self.Li[:, self.N :, : self.N].expand((batch_size, 3, self.N)), Q_x)
        A_y = torch.bmm(self.Li[:, self.N :, : self.N].expand((batch_size, 3, self.N)), Q_y)

        A_x = A_x.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, h, w, 1, 1)
        A_y = A_y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, h, w, 1, 1)
        
        points_x_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        if b == 1:
            delta_x = points_x_for_summation - P_x
            delta_y = points_y_for_summation - P_y
        else:
            delta_x = points_x_for_summation - P_x.expand_as(points_x_for_summation)
            delta_y = points_y_for_summation - P_y.exasnd_as(points_y_for_summation)

        dist_squared = torch.pow(delta_x, 2) + torch.pow(delta_y, 2)
        dist_squared[dist_squared == 0] = 1

        U = torch.mul(dist_squared, torch.log(dist_squared))

        points_x_batch = points[:, :, :, 0].unsqueeze(3)
        points_y_batch = points[:, :, :, 1].unsqueeze(3)

        if b == 1:
            points_x_batch = points_x_batch.expand((batch_size, ) + points_x_batch.size()[1:])
            points_y_batch = points_y_batch.expand((batch_size, ) + points_x_batch.size()[1:])

        points_x_prime = A_x[:, :, :, :, 0] + \
                        torch.mul(A_x[:, :, :, :, 1], points_x_batch) + \
                        torch.mul(A_x[:, :, :, :, 2], points_y_batch) + \
                        torch.sum(torch.mul(W_x, U.expand_as(W_x)), 4)

        points_y_prime = A_y[:, :, :, :, 0] + \
                        torch.mul(A_y[:, :, :, :, 1], points_x_batch) + \
                        torch.mul(A_y[:, :, :, :, 2], points_y_batch) + \
                        torch.sum(torch.mul(W_y, U.expand_as(W_y)), 4)

        return torch.cat((points_x_prime, points_y_prime), 3)

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_x, self.grid_y), 3))
        return warped_grid

if __name__ == "__main__":
    feature1 = torch.randn(4, 1, 256, 192)
    feature2 = torch.randn(4, 1, 256, 192)
    
    feature1 = FeatureExtraction(1)(feature1)
    feature1 = feature_norm("L2")(feature1)
    
    feature2 = FeatureExtraction(1)(feature2)
    feature2 = feature_norm("L2")(feature2)

    correlation = FeatureCorrelation()(feature1, feature2)
    
    theta = FeatureRegression(in_channels=192, out_feature=50)(correlation)
    TpsGridGen()