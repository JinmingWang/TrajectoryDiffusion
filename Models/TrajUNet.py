import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *


class EmbedBlock(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, max_time: int, feature_dim: int = 128) -> None:
        super().__init__()

        position = torch.arange(max_time, dtype=torch.float32, device=self.device).unsqueeze(1)    # (max_time, 1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2, dtype=torch.float32, device=self.device) * -(math.log(1.0e4) / feature_dim))    # (feature_dim / 2)
        self.pos_enc = torch.zeros((max_time, feature_dim), dtype=torch.float32, device=self.device)    # (max_time, feature_dim)
        self.pos_enc[:, 0::2] = torch.sin(position * div_term)
        self.pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.time_embed_layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        self.attr_embed_layers = nn.Sequential(
            nn.Linear(3, feature_dim),  # 3 for travel distance, avg move distance, departure time
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    
    def forward(self, time: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embed_layers(self.pos_enc[time, :])    # (B, feature_dim)
        attr_embed = self.attr_embed_layers(attr)    # (B, feature_dim)
        return torch.cat([time_embed, attr_embed], dim=1).unsqueeze(2)    # (B, feature_dim*2, 1)


class ResnetBlock(nn.Module):
    def __init__(self, in_c: int, norm_groups: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # input: (B, in_c, L)
        self.stage1 = nn.Sequential(
            nn.GroupNorm(norm_groups, in_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_c, embed_dim, 3, padding=1),    # (B, in_c, L)
        )

        self.stage2 = nn.Sequential(
            nn.GroupNorm(norm_groups, embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, in_c, 3, padding=1),    # (B, in_c, L)
        )

    def forward(self, x: torch.Tensor, time_attr_embed: torch.Tensor) -> None:
        """
        :param x: (B, in_c, L)
        :param time_attr_embed: (B, 256, 1)
        :return: (B, in_c, L)
        """
        return x + self.stage2(self.stage1(x) + time_attr_embed)
    

class MidAttnBlock(nn.Module):
    def __init__(self, in_c: int, norm_groups: int) -> None:
        super().__init__()

        self.res1 = ResnetBlock(in_c, norm_groups)

        self.linear = nn.Conv1d(in_c, in_c*3, 1, 1, 0)    # (B, in_c*3, L)

        self.res2 = ResnetBlock(in_c, norm_groups)

    
    def forward(self, x: torch.Tensor, time_attr_embed: torch.Tensor) -> None:
        # input: (B, in_c, L)
        x = self.res1(x, time_attr_embed)
        kt, qt, vt = torch.split(self.linear(x), x.shape[1], dim=1)    # (B, in_c, L) * 3
        attn = torch.softmax((qt.transpose(1, 2) @ kt) / (x.shape[1] ** 0.5), dim=2)    # (B, L, L)
        x = self.res2((attn @ vt.transpose(1, 2)).transpose(1, 2), time_attr_embed)    # (B, in_c, L)
        return x
    

class TrajUNet(nn.Module):
    def __init__(self, stem_channels: int, diffusion_steps: int, sampling_blocks: int, res_blocks: int) -> None:
        super().__init__()

        self.sampling_blocks = sampling_blocks
        self.res_blocks = res_blocks

        self.embed_block = EmbedBlock(diffusion_steps, 128)

        self.stem = nn.Conv1d(2, stem_channels, 3, 1, 1)    # (B, stem_channels, L)

        self.down_blocks = nn.ModuleList([self.__makeEncoderStage(stem_channels * 2 ** i) for i in range(sampling_blocks)])

        self.mid_attn_block = MidAttnBlock(stem_channels * 2 ** sampling_blocks, 32)

        self.up_blocks = nn.ModuleList([self.__makeDecoderStage(stem_channels * 2 ** (sampling_blocks - i)) for i in range(sampling_blocks)])

        self.head = nn.Conv1d(stem_channels, 2, 3, 1, 1)    # (B, 2, L)


    def __makeEncoderStage(self, channels: int, expand_ratio: int = 2) -> nn.ModuleList:
        layers = []
        for i in range(self.res_blocks):
            layers.append(ResnetBlock(channels, 1))
        layers.append(nn.Conv1d(channels, channels*expand_ratio, 3, 2, 1))     # downsample
        return nn.ModuleList(layers)
    

    def __makeDecoderStage(self, channels: int, shrink_ratio: int = 2) -> nn.ModuleList:
        layers = []
        layers.append(nn.Conv1d(channels*2, channels, 3, 1, 1))   # fuse
        for i in range(self.res_blocks):
            layers.append(ResnetBlock(channels, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))    # upsample
        layers.append(nn.Conv1d(channels, channels//shrink_ratio, 3, 1, 1))    # shrink
        return nn.ModuleList(layers)
    

    def __encoderForward(self, x: torch.Tensor, time_attr_embed: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: (B, stem_channels, L)
        :param time_attr_embed: (B, 256, 1)
        :return: List of (B, C', L//2**i)
        """

        outputs = []
        for down_stage in self.down_blocks:
            for layer in down_stage[:-1]:
                x = layer(x, time_attr_embed)    # (B, C, L) -> (B, C, L)
            x = down_stage[-1](x)   # downsample
            outputs.append(x)
        return outputs
    

    def __decoderForward(self, x: torch.Tensor, time_attr_embed: torch.Tensor, down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param time_attr_embed: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i-1]], dim=1)   # (B, C*2, L//2**i)
            x = up_stage[0](x)
            for layer in up_stage[1:-2]:
                x = layer(x, time_attr_embed)
            x = up_stage[-2](x)
            x = up_stage[-1](x)
        return x
    

    def forward(self, x: torch.Tensor, time: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 2, L)
        :param time: (B, )
        :param attr: (B, 3)
        :return: (B, 2, L)
        """
        time_attr_embed = self.embed_block(time, attr)    # (B, 256, 1)
        x = self.stem(x)    # (B, stem_channels, L)
        down_outputs = self.__encoderForward(x, time_attr_embed)    # List of (B, C', L//2**i)
        x = self.mid_attn_block(down_outputs[-1], time_attr_embed)    # (B, C', L//2**i)
        x = self.__decoderForward(x, time_attr_embed, down_outputs)    # (B, C, L)
        return self.head(x)
    

if __name__ == "__main__":
    model = TrajUNet(stem_channels=32, diffusion_steps=300, sampling_blocks=4, res_blocks=2).cuda()
    x = torch.randn(1, 2, 128).cuda()
    time = torch.tensor([0,], dtype=torch.long, device='cuda')
    attr = torch.randn(1, 3).cuda()
    y = model(x, time, attr)
    print(y.shape)



