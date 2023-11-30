import torch

from Models.ModelUtils import *
from math import sqrt


class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        """Channel attention module. Expect input shape (B, C, L).

        :param channels: input and output channels
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels//2, 1, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv1d(channels//2, channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        return x * self.layers(x)   # (B, C, L)


class ParallelDilationRes(nn.Module):
    def __init__(self, channels: int):
        """Parallel dilated convolution with 4 branches. Expect input shape (B, C, L).

        This module is used to capture multi-scale dependencies. Similar structure can be found in DeepLabV3+ (a image segmentation model)

        :param channels: input and output channels
        """
        super().__init__()

        # (B, C, L)
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, 5, 1, 2, dilation=1, bias=False),  # receptive field: 5
            nn.Conv1d(channels, channels, 5, 1, 4, dilation=2, bias=False),  # receptive field: 9
            nn.Conv1d(channels, channels, 5, 1, 6, dilation=3, bias=False),  # receptive field: 13
        ])

        self.head = nn.Sequential(
            nn.BatchNorm1d(channels * 3),
            Swish(),
            ChannelAttention(channels * 3),
            nn.Conv1d(channels * 3, channels, 1, 1, 0, bias=False),
        )

        self.act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        branch_results = torch.cat([branch(x) for branch in self.branches], dim=1)     # (B, C * 3, L)
        return self.act(x + self.head(branch_results))     # (B, C, L)


class SEResBlock(nn.Module):
    def __init__(self, channels: int, expand: int = 2):
        super().__init__()

        mid_c = channels * expand

        # Recepetive field: 5 + 4 = 9
        self.layers = nn.Sequential(
            ConvNormAct(channels, mid_c, 5, 1, 2),
            ConvNormAct(mid_c, mid_c, 5, 1, 2),
            ChannelAttention(mid_c),
            nn.Conv1d(mid_c, channels, 1, 1, 0)
        )

        self.final_act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_act(x + self.layers(x))
    

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)
    

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        return self.attn(x, x, x)[0].transpose(-1, -2)



class InvTransformerBlock(nn.Module):
    def __init__(self, in_c: int, in_seq_len: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # (B, C, L)

        # attn computes inter-channel attention
        self.attn = nn.MultiheadAttention(embed_dim=in_seq_len, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.layers = nn.Sequential(
            ConvNormAct(in_c, in_c*2, 3, 1, 1),
            nn.Conv1d(in_c*2, in_c, 3, 1, 1)
        )

        self.final_act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.attn(x, x, x)[0] + x   # (B, C, L)
        return self.final_act(x + self.layers(x))   # (B, C, L)


