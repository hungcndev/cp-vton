import torch
import torch.nn as nn

from typing import Optional, List

# -------------------------------------------------------------------------- #

class FeatureExtraction(nn.Moudle):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[List[int]] = None
       ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [6]
        layers = []

        
    def forward(self, x):

        return x