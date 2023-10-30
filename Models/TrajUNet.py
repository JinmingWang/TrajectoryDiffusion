from Models.TrajUNetBlocks import *

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
        self.embed_block = EmbedBlock(diffusion_steps, 128)

        # Create First layer for UNet
        self.stem = nn.Conv1d(2, channel_schedule[0], 3, 1, 1)    # (B, stem_channels, L)

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        traj_lengths = [traj_length]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i]))
            traj_lengths.append(traj_lengths[-1] // 2)

        # Create Middle Attention Block for UNet
        self.mid_attn_block = AttnBlock(out_channels[-1], norm_groups=32)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = channel_schedule[-1:0:-1]
        out_channels = channel_schedule[-2::-1]
        upsample_targets = traj_lengths[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(self.__makeDecoderStage(in_channels[i] * 2, out_channels[i], upsample_targets[i]))

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
    

    def __makeDecoderStage(self, in_c: int, out_c: int, upsample_size: int) -> nn.ModuleList:
        layers = [ResnetBlock(in_c, out_c, norm_groups=32)]     # fuse with skip connection
        for i in range(self.res_blocks - 1):
            layers.append(ResnetBlock(out_c, norm_groups=32))
        layers.append(nn.Upsample(size=upsample_size, mode='nearest'))    # upsample
        layers.append(nn.Conv1d(out_c, out_c, 3, 1, 1))    # shrink
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
            for layer in up_stage[:-2]:
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
    model = TrajUNet(channel_schedule=[128, 128, 256, 512, 1024], traj_length=200, diffusion_steps=300, res_blocks=2).cuda()
    x = torch.randn(1, 2, 200).cuda()
    time = torch.tensor([0,], dtype=torch.long, device='cuda')
    attr = torch.randn(1, 3).cuda()
    y = model(x, time, attr)
    print(y.shape)



