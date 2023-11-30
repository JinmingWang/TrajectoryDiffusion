from Models.StableDiffusionBlocks import *


class SDUNet(nn.Module):
    def __init__(self, channel_schedule: List[int], diffusion_steps: int = 300,
                 res_blocks: int = 2) -> None:
        super().__init__()

        self.channel_schedule = channel_schedule
        self.stages = len(channel_schedule) - 1
        self.res_blocks = res_blocks

        # Time and Attribute Embedding
        self.embed_block = ConditionMixBlock(diffusion_steps, embed_dim=128)

        # Create First layer for UNet
        self.stem = nn.Conv1d(4, channel_schedule[0], 3, 1, 1)  # (B, stem_channels, L)

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i]))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = CrossAttention(embed_dim=out_channels[-1], hidden_dim=out_channels[-1] * 2, out_dim=out_channels[-1], num_heads=4)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = channel_schedule[-1:0:-1]
        out_channels = channel_schedule[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(self.__makeDecoderStage(in_channels[i] * 2, out_channels[i]))

        # Create last for UNet
        self.head = nn.Sequential(
            nn.GroupNorm(32, channel_schedule[0]),
            nn.Conv1d(channel_schedule[0], 4, 3, 1, 1)  # (B, 2, L)
        )


    def __makeEncoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [CrossAttention(embed_dim=in_c, hidden_dim=in_c*2, out_dim=out_c, num_heads=4),
                  CrossAttention(embed_dim=out_c, hidden_dim=out_c*2, out_dim=out_c, num_heads=4, downscale=True)]

        for i in range(self.res_blocks - 2):
            layers.insert(1, CrossAttention(embed_dim=out_c, hidden_dim=out_c*2, out_dim=out_c, num_heads=4))
        return nn.ModuleList(layers)


    def __makeDecoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [CrossAttention(embed_dim=in_c, hidden_dim=in_c * 2, out_dim=out_c, num_heads=4),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  CrossAttention(embed_dim=out_c, hidden_dim=out_c * 2, out_dim=out_c, num_heads=4)]

        for i in range(self.res_blocks - 2):
            layers.insert(1, CrossAttention(embed_dim=out_c, hidden_dim=out_c * 2, out_dim=out_c, num_heads=4))
        return nn.ModuleList(layers)


    def __encoderForward(self, x: torch.Tensor, embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: (B, stem_channels, L)
        :param embedding: (B, 256, 1)
        :return: List of (B, C', L//2**i)
        """

        outputs = []
        for down_stage in self.down_blocks:
            for layer in down_stage:
                x = layer(x, embedding)  # (B, C, L) -> (B, C, L)
            outputs.append(x)
        return outputs


    def __decoderForward(self, x: torch.Tensor, embedding: torch.Tensor,
                         down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param embedding: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)  # (B, C*2, L//2**i)
            for layer in up_stage[:-2]:
                x = layer(x, embedding)
            x = up_stage[-2](x)
            x = up_stage[-1](x, embedding)
        return x


    def forward(self, x: torch.Tensor, time: torch.Tensor, cat_attr: torch.Tensor,
                num_attr: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 4, L)
        :param time: (B, )
        :param cat_attr: (B, 5)
        :param num_attr: (B, 3)
        :return: (B, 4, L)
        """
        embedding = self.embed_block(time, cat_attr, num_attr)  # (B, C, 7)
        x = self.stem(x)  # (B, stem_channels, L)
        down_outputs = self.__encoderForward(x, embedding)  # List of (B, C', L//2**i)
        x = self.mid_attn_block(down_outputs[-1], embedding)  # (B, C', L//2**i)
        x = self.__decoderForward(x, embedding, down_outputs)  # (B, C, L)
        return self.head(x)


# 24.221647 ms
class SDUNet_CAR(nn.Module):
    def __init__(self, traj_dim: int, channel_schedule: List[int], diffusion_steps: int = 300,
                 res_blocks: int = 2) -> None:
        super().__init__()

        self.channel_schedule = channel_schedule
        self.stages = len(channel_schedule) - 1
        self.res_blocks = res_blocks

        # Time and Attribute Embedding
        self.embed_block = ConditionMixBlock(diffusion_steps, hidden_dim=128, embed_dim=128)

        # Create First layer for UNet
        self.stem = nn.Conv1d(traj_dim, channel_schedule[0], 3, 1, 1)  # (B, stem_channels, L)

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i]))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = CrossAttentionRes(embed_dim=out_channels[-1], out_dim=out_channels[-1], num_heads=8)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = channel_schedule[-1:0:-1]
        out_channels = channel_schedule[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(self.__makeDecoderStage(in_channels[i] * 2, out_channels[i]))

        # Create last for UNet
        self.head = nn.Sequential(
            nn.GroupNorm(32, channel_schedule[0]),
            nn.Conv1d(channel_schedule[0], traj_dim, 3, 1, 1)  # (B, 2, L)
        )


    def __makeEncoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [CrossAttentionRes(embed_dim=in_c, out_dim=out_c, num_heads=8),
                  CrossAttentionRes(embed_dim=out_c, out_dim=out_c, num_heads=8),
                  nn.Conv1d(out_c, out_c, 3, 2, 1)]

        for i in range(self.res_blocks - 2):
            layers.insert(1, CrossAttentionRes(embed_dim=out_c, out_dim=out_c, num_heads=8))
        return nn.ModuleList(layers)


    def __makeDecoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [CrossAttentionRes(embed_dim=in_c, out_dim=out_c, num_heads=8),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  CrossAttentionRes(embed_dim=out_c, out_dim=out_c, num_heads=8)]

        for i in range(self.res_blocks - 2):
            layers.insert(1, CrossAttentionRes(embed_dim=out_c, out_dim=out_c, num_heads=8))
        return nn.ModuleList(layers)


    def __encoderForward(self, x: torch.Tensor, embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: (B, stem_channels, L)
        :param embedding: (B, 256, 1)
        :return: List of (B, C', L//2**i)
        """

        outputs = []
        for down_stage in self.down_blocks:
            for layer in down_stage[:-1]:
                x = layer(x, embedding)  # (B, C, L) -> (B, C, L)
            x = down_stage[-1](x)  # (B, C, L) -> (B, C', L//2)
            outputs.append(x)
        return outputs


    def __decoderForward(self, x: torch.Tensor, embedding: torch.Tensor,
                         down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param embedding: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)  # (B, C*2, L//2**i)
            for layer in up_stage[:-2]:
                x = layer(x, embedding)
            x = up_stage[-2](x)
            x = up_stage[-1](x, embedding)
        return x


    def forward(self, x: torch.Tensor, time: torch.Tensor, cat_attr: torch.Tensor,
                num_attr: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 4, L)
        :param time: (B, )
        :param cat_attr: (B, 5)
        :param num_attr: (B, 3)
        :return: (B, 4, L)
        """
        embedding = self.embed_block(time, cat_attr, num_attr)  # (B, C, 7)
        x = self.stem(x)  # (B, stem_channels, L)
        down_outputs = self.__encoderForward(x, embedding)  # List of (B, C', L//2**i)
        x = self.mid_attn_block(down_outputs[-1], embedding)  # (B, C', L//2**i)
        x = self.__decoderForward(x, embedding, down_outputs)  # (B, C, L)
        return self.head(x)


# 35.810380 ms
# 4
# [64, 128, 128, 256]
# res_blocks: 4
# c_reduce: 2
class SDUNet_InvTrans(nn.Module):
    def __init__(self, traj_dim: int, channel_schedule: List[int], diffusion_steps: int = 300, res_blocks: int = 2,
                 c_reduce=2) -> None:
        super().__init__()

        self.channel_schedule = channel_schedule
        self.stages = len(channel_schedule) - 1
        self.res_blocks = res_blocks
        self.c_reduce = c_reduce

        # Time and Attribute Embedding
        self.embed_block = ConditionMixBlock(diffusion_steps, hidden_dim=256, embed_dim=128)

        # Create First layer for UNet
        self.stem = nn.Conv1d(traj_dim, channel_schedule[0], 3, 1, 1)  # (B, stem_channels, L)

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i]))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = CrossAttentionRes(embed_dim=out_channels[-1], out_dim=out_channels[-1], num_heads=8)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = channel_schedule[-1:0:-1]
        out_channels = channel_schedule[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(self.__makeDecoderStage(in_channels[i] * 2, out_channels[i]))

        # Create last for UNet
        self.head = nn.Sequential(
            nn.GroupNorm(32, channel_schedule[0]),
            nn.Conv1d(channel_schedule[0], traj_dim, 3, 1, 1)  # (B, 2, L)
        )


    def __makeEncoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [ConvInvTransRes(embed_dim=in_c, out_dim=out_c, num_heads=4, c_reduce=self.c_reduce),
                  nn.Conv1d(out_c, out_c, 3, 2, 1)]

        for i in range(self.res_blocks - 1):
            layers.insert(1, ConvInvTransRes(embed_dim=out_c, out_dim=out_c, num_heads=4, c_reduce=self.c_reduce))
        return nn.ModuleList(layers)


    def __makeDecoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [ConvInvTransRes(embed_dim=in_c, out_dim=out_c, num_heads=4, c_reduce=self.c_reduce),
                  nn.Conv1d(out_c, out_c, 3, 1, 1)]

        for i in range(self.res_blocks - 1):
            layers.insert(1, ConvInvTransRes(embed_dim=out_c, out_dim=out_c, num_heads=4, c_reduce=self.c_reduce))
        return nn.ModuleList(layers)


    def __encoderForward(self, x: torch.Tensor, embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: (B, stem_channels, L)
        :param embedding: (B, 256, 1)
        :return: List of (B, C', L//2**i)
        """

        outputs = []
        for down_stage in self.down_blocks:
            for layer in down_stage[:-1]:
                x = layer(x, embedding)  # (B, C, L) -> (B, C, L)
            x = down_stage[-1](x)  # (B, C, L) -> (B, C', L//2)
            outputs.append(x)
        return outputs


    def __decoderForward(self, x: torch.Tensor, embedding: torch.Tensor,
                         down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param embedding: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)  # (B, C*2, L//2**i)
            for layer in up_stage[:-1]:
                x = layer(x, embedding)
            # upsample
            x = F.interpolate(x, size=down_outputs[-i - 2].shape[-1], mode='nearest')
            x = up_stage[-1](x)
        return x


    def forward(self, x: torch.Tensor, time: torch.Tensor, cat_attr: torch.Tensor,
                num_attr: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 4, L)
        :param time: (B, )
        :param cat_attr: (B, 5)
        :param num_attr: (B, 3)
        :return: (B, 4, L)
        """
        embedding = self.embed_block(time, cat_attr, num_attr)  # (B, C, 7)
        x = self.stem(x)  # (B, stem_channels, L)
        down_outputs = self.__encoderForward(x, embedding)  # List of (B, C', L//2**i)
        down_outputs.insert(0, x)
        x = self.mid_attn_block(down_outputs[-1], embedding)  # (B, C', L//2**i)
        x = self.__decoderForward(x, embedding, down_outputs)  # (B, C, L)
        return self.head(x)


class SDUNet_Trans(nn.Module):
    def __init__(self, traj_dim: int, channel_schedule: List[int], diffusion_steps: int = 300,
                 res_blocks: int = 2) -> None:
        super().__init__()

        self.channel_schedule = channel_schedule
        self.stages = len(channel_schedule) - 1
        self.res_blocks = res_blocks

        # Time and Attribute Embedding
        self.embed_block = ConditionMixBlock(diffusion_steps, hidden_dim=256, embed_dim=128)

        # Create First layer for UNet
        self.stem = nn.Conv1d(traj_dim, channel_schedule[0], 3, 1, 1)  # (B, stem_channels, L)

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i]))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = CrossAttentionRes(embed_dim=out_channels[-1], out_dim=out_channels[-1], num_heads=8)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = channel_schedule[-1:0:-1]
        out_channels = channel_schedule[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(self.__makeDecoderStage(in_channels[i] * 2, out_channels[i]))

        # Create last for UNet
        self.head = nn.Sequential(
            nn.GroupNorm(32, channel_schedule[0]),
            nn.Conv1d(channel_schedule[0], traj_dim, 3, 1, 1)  # (B, 2, L)
        )


    def __makeEncoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [TransRes(embed_dim=in_c, out_dim=out_c, num_heads=4),
                  nn.Conv1d(out_c, out_c, 3, 2, 1)]

        for i in range(self.res_blocks - 1):
            layers.insert(1, TransRes(embed_dim=out_c, out_dim=out_c, num_heads=4))
        return nn.ModuleList(layers)


    def __makeDecoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [TransRes(embed_dim=in_c, out_dim=out_c, num_heads=4),
                  nn.Conv1d(out_c, out_c, 3, 1, 1)]

        for i in range(self.res_blocks - 1):
            layers.insert(1, TransRes(embed_dim=out_c, out_dim=out_c, num_heads=4))
        return nn.ModuleList(layers)


    def __encoderForward(self, x: torch.Tensor, embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: (B, stem_channels, L)
        :param embedding: (B, 256, 1)
        :return: List of (B, C', L//2**i)
        """

        outputs = []
        for down_stage in self.down_blocks:
            for layer in down_stage[:-1]:
                x = layer(x, embedding)  # (B, C, L) -> (B, C, L)
            x = down_stage[-1](x)  # (B, C, L) -> (B, C', L//2)
            outputs.append(x)
        return outputs


    def __decoderForward(self, x: torch.Tensor, embedding: torch.Tensor,
                         down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param embedding: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)  # (B, C*2, L//2**i)
            for layer in up_stage[:-1]:
                x = layer(x, embedding)
            # upsample
            x = F.interpolate(x, size=down_outputs[-i - 2].shape[-1], mode='nearest')
            x = up_stage[-1](x)
        return x


    def forward(self, x: torch.Tensor, time: torch.Tensor, cat_attr: torch.Tensor,
                num_attr: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 4, L)
        :param time: (B, )
        :param cat_attr: (B, 5)
        :param num_attr: (B, 3)
        :return: (B, 4, L)
        """
        embedding = self.embed_block(time, cat_attr, num_attr)  # (B, C, 7)
        x = self.stem(x)  # (B, stem_channels, L)
        down_outputs = self.__encoderForward(x, embedding)  # List of (B, C', L//2**i)
        down_outputs.insert(0, x)
        x = self.mid_attn_block(down_outputs[-1], embedding)  # (B, C', L//2**i)
        x = self.__decoderForward(x, embedding, down_outputs)  # (B, C, L)
        return self.head(x)


if __name__ == "__main__":
    # 16.579784 ms for 4 heads
    # 20.003283 ms for 8 heads
    # 16.453849 ms for 8 head [64, 128, 256, 512]
    model = SDUNet_Trans(4, channel_schedule=[128, 256, 512, 1024], diffusion_steps=300, res_blocks=2).cuda()

    x = torch.randn(1, 4, 25).cuda()
    time = torch.tensor([0, ], dtype=torch.long, device='cuda')
    cat_attr = torch.tensor([[0, 0, 0, 0, 0], ], dtype=torch.long, device='cuda')
    num_attr = torch.tensor([[0, 0, 0], ], dtype=torch.float, device='cuda')
    y = model(x, time, cat_attr, num_attr)
    print(y.shape)

    inferSpeedTest1K(model, x, time, cat_attr, num_attr)