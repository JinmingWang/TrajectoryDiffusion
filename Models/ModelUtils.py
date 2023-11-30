import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from tqdm import tqdm
import time

class Swish(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    

class ConvNormAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int=0, d: int=1, g: int=1):
        """Convolution 1D with BatchNorm1d and LeakyReLU activation. Expect input shape (B, C, L).

        :param in_c: in channels
        :param out_c: out channels
        :param k: kernel size
        :param s: stride
        :param p: padding, defaults to 0
        :param d: dilation, defaults to 1
        :param g: group, defaults to 1
        """
        super(ConvNormAct, self).__init__(
            nn.Conv1d(in_c, out_c, k, s, p, d, g, bias=False),
            nn.BatchNorm1d(out_c),
            Swish()
        )


def inferSpeedTest1K(model, *dummy_inputs):
    model.eval()
    start = time.time()
    with torch.no_grad():
        if len(dummy_inputs) == 1:
            for i in tqdm(range(1000)):
                model(dummy_inputs[0])
        else:
            for i in tqdm(range(1000)):
                model(*dummy_inputs)
    end = time.time()
    print(f"Each inference takes {end - start:.6f} ms")
