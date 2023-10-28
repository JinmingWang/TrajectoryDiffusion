from Models.ModelUtils import *
import math


class EmbedBlock(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self, max_time: int, feature_dim: int = 128) -> None:
        super().__init__()

        position = torch.arange(max_time, dtype=torch.float32, device=self.device).unsqueeze(1)  # (max_time, 1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2, dtype=torch.float32, device=self.device) * -(
                    math.log(1.0e4) / feature_dim))  # (feature_dim / 2)
        self.pos_enc = torch.zeros((max_time, feature_dim), dtype=torch.float32,
                                   device=self.device)  # (max_time, feature_dim)
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
        time_embed = self.time_embed_layers(self.pos_enc[time, :])  # (B, feature_dim)
        attr_embed = self.attr_embed_layers(attr)  # (B, feature_dim)
        return torch.cat([time_embed, attr_embed], dim=1).unsqueeze(2)  # (B, feature_dim*2, 1)


class ResnetBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int = None, norm_groups: int = 32, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        if out_c is None:
            out_c = in_c

        # input: (B, in_c, L)
        self.stage1 = nn.Sequential(
            nn.GroupNorm(norm_groups, in_c, eps=1e-6),
            Swish(),
            nn.Conv1d(in_c, out_c, 3, padding=1),  # (B, in_c, L)
        )

        self.embed_proj = nn.Sequential(
            Swish(),
            nn.Conv1d(embed_dim, out_c, 1, padding=0),  # (B, embed_dim, L)
        )

        self.stage2 = nn.Sequential(
            nn.GroupNorm(norm_groups, out_c, eps=1e-6),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, 3, padding=1),  # (B, in_c, L)
        )

        self.shortcut = nn.Identity() if in_c == out_c else nn.Conv1d(in_c, out_c, 1, padding=0)  # (B, out_c, L)


    def forward(self, x: torch.Tensor, time_attr_embed: torch.Tensor) -> None:
        """
        :param x: (B, in_c, L)
        :param time_attr_embed: (B, 256, 1)
        :return: (B, in_c, L)
        """
        embed = self.embed_proj(time_attr_embed)  # (B, in_c, L)
        return self.shortcut(x) + self.stage2(self.stage1(x) + embed)  # (B, in_c, L)


class AttnBlock(nn.Module):
    def __init__(self, in_c: int, norm_groups: int) -> None:
        super().__init__()

        self.res1 = ResnetBlock(in_c, in_c, norm_groups)

        self.qkv_getter = nn.Sequential(
            nn.GroupNorm(norm_groups, in_c, eps=1e-6),
            nn.Conv1d(in_c, in_c * 3, 1, 1, 0),  # (B, in_c*3, L)
        )
        self.project_out = nn.Conv1d(in_c, in_c, 1, 1, 0)  # (B, in_c, L)

        self.res2 = ResnetBlock(in_c, in_c, norm_groups)


    def forward(self, x: torch.Tensor, time_attr_embed: torch.Tensor) -> None:
        # input: (B, in_c, L)
        x = self.res1(x, time_attr_embed)

        q, k, v = torch.split(self.qkv_getter(x).transpose(1, 2), x.shape[1], dim=2)  # (B, L, in_c) * 3
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (x.shape[1] ** 0.5), dim=2)  # (B, L, L)
        x = x + self.project_out(torch.bmm(attn, v).transpose(1, 2))  # (B, in_c, L)

        return self.res2(x, time_attr_embed)