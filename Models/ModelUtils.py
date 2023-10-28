import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class Swish(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)