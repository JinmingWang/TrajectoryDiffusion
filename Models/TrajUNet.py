from Models.TrajUNetBlocks import *

# 11.169695 ms
class TrajUNet(nn.Module):
    # stem_channels: 128
    # diffusion_steps: 300
    # resolution ratios: 1, 1, 2, 2, 2
    # resolutions: 2 --stem-> 128 --down-> 128 --down-> 256 --down-> 512 --down-> 1024
    # sampling_blocks: 4
    def __init__(self, channel_schedule: List[int], traj_length: int = 200, diffusion_steps: int = 300, res_blocks: int = 2) -> None:
        super().__init__()

        self.channel_schedule = channel_schedule
        self.stages = len(channel_schedule) - 1
        self.res_blocks = res_blocks

        # Time and Attribute Embedding
        self.embed_block = WideAndDeepEmbedBlock(diffusion_steps, hidden_dim=256, embed_dim=128)

        # Create First layer for UNet
        self.stem = nn.Conv1d(2, channel_schedule[0], 3, 1, 1)    # (B, stem_channels, L)

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i]))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = AttnBlock(out_channels[-1], norm_groups=32)

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
            nn.Conv1d(channel_schedule[0], 2, 3, 1, 1)  # (B, 2, L)
        )


    def __makeEncoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [ResnetBlock(in_c, out_c, norm_groups=32)]
        for i in range(self.res_blocks - 1):
            layers.append(ResnetBlock(out_c, norm_groups=32))
        layers.append(nn.Conv1d(out_c, out_c, 3, 2, 1))     # downsample
        return nn.ModuleList(layers)
    

    def __makeDecoderStage(self, in_c: int, out_c: int) -> nn.ModuleList:
        layers = [ResnetBlock(in_c, out_c, norm_groups=32)]     # fuse with skip connection
        for i in range(self.res_blocks - 1):
            layers.append(ResnetBlock(out_c, norm_groups=32))
        # layers.append(nn.Upsample(size=upsample_size, mode='nearest'))    # upsample
        layers.append(nn.Conv1d(out_c, out_c, 3, 1, 1))    # shrink
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
                x = layer(x, embedding)    # (B, C, L) -> (B, C, L)
            x = down_stage[-1](x)   # downsample
            outputs.append(x)
        return outputs
    

    def __decoderForward(self, x: torch.Tensor, embedding: torch.Tensor, down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param embedding: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i-1]], dim=1)   # (B, C*2, L//2**i)
            for layer in up_stage[:-2]:
                x = layer(x, embedding)
            x = F.interpolate(x, size=down_outputs[-i - 2].shape[-1], mode='nearest')
            x = up_stage[-1](x)
        return x
    

    def forward(self, x: torch.Tensor, time: torch.Tensor, cat_attr: torch.Tensor, num_attr: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 2, L)
        :param time: (B, )
        :param cat_attr: (B, 5)
        :param num_attr: (B, 3)
        :return: (B, 2, L)
        """
        embedding = self.embed_block(time, cat_attr, num_attr)    # (B, 256, 1)
        x = self.stem(x)    # (B, stem_channels, L)
        down_outputs = self.__encoderForward(x, embedding)    # List of (B, C', L//2**i)
        down_outputs.insert(0, x)
        x = self.mid_attn_block(down_outputs[-1], embedding)    # (B, C', L//2**i)
        x = self.__decoderForward(x, embedding, down_outputs)    # (B, C, L)
        return self.head(x)


if __name__ == "__main__":
    model = TrajUNet(channel_schedule=[128, 128, 256, 512, 1024], diffusion_steps=500, res_blocks=2).cuda()
    # for i in range(100, 210, 7):
    #     x = torch.randn(1, 2, i).cuda()
    #     time = torch.tensor([0,], dtype=torch.long, device='cuda')
    #     cat_attr = torch.tensor([[0, 0, 0, 0, 0],], dtype=torch.long, device='cuda')
    #     num_attr = torch.tensor([[0, 0, 0],], dtype=torch.float, device='cuda')
    #     y = model(x, time, cat_attr, num_attr)
    #     print(y.shape)

    # 10.708994 ms

    x = torch.randn(1, 2, 256).cuda()
    time = torch.tensor([0, ], dtype=torch.long, device='cuda')
    cat_attr = torch.tensor([[0, 0, 0, 0, 0], ], dtype=torch.long, device='cuda')
    num_attr = torch.tensor([[0, 0, 0], ], dtype=torch.float, device='cuda')
    inferSpeedTest1K(model, x, time, cat_attr, num_attr)



