from Models.ModelUtils import *
import math


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, out_dim: int, num_heads: int, downscale: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.convs = nn.Sequential(
            ConvNormAct(embed_dim, hidden_dim, 3, 1, 1),
            nn.Conv1d(hidden_dim, embed_dim, 3, 1, 1),
        )

        self.cond_proj = nn.Linear(128, embed_dim)

        self.out = nn.Sequential(
            Swish(),
            nn.Conv1d(embed_dim, out_dim, 3, 2 if downscale else 1, 1),
            Swish(),
        )


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # cond: (B, C, N)
        x = x.transpose(1, 2)  # (B, L, C)
        cond = self.cond_proj(cond.transpose(1, 2))  # (B, N, C)
        # Q: (B, L, C) -> (B, L, embed_dim)
        # K, V: (B, N, C) -> (B, N, embed_dim)
        # Q * K^T: (B, L, embed_dim) * (B, embed_dim, N) -> (B, L, N)
        # attn: (B, L, N) * (B, N, embed_dim) -> (B, L, embed_dim)
        x = self.multihead_attn(x, cond, cond)[0] + x  # (B, L, embed_dim)

        x = x.transpose(1, 2)  # (B, embed_dim, L)
        x = self.convs(x) + x  # (B, embed_dim, L)

        return self.out(x)  # (B, out_dim, L)
    

class CrossAttentionRes(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int, dropout: float = 0.1, norm_groups: int = 32):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.GroupNorm(norm_groups, embed_dim, eps=1e-6),
            Swish(),
            nn.Conv1d(embed_dim, out_dim, 3, padding=1),  # (B, in_c, L)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(7 * 128, 7 * out_dim),
            nn.Unflatten(1, (7, -1)), # (B, 7, 128)
        )

        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)

        self.stage2 = nn.Sequential(
            nn.GroupNorm(norm_groups, out_dim, eps=1e-6),
            Swish(),
            nn.Conv1d(out_dim, out_dim, 3, padding=1),  # (B, in_c, L)
        )

        self.shortcut = nn.Identity() if embed_dim == out_dim else nn.Conv1d(embed_dim, out_dim, 1, padding=0)  # (B, out_c, L)


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dim, L)
        # cond: (B, embed_dim, N)
        identity = self.shortcut(x)  # (B, out_dim, L)

        x = self.stage1(x).transpose(1, 2)  # (B, L, out_dim)
        cond = self.cond_proj(cond)  # (B, N, out_dim)

        x = self.multihead_attn(x, cond, cond)[0] + x  # (B, L, out_dim)

        x = x.transpose(1, 2)  # (B, out_dim, L)
        x = self.stage2(x)  # (B, out_dim, L)

        return identity + x  # (B, out_dim, L)


class ConditionMixBlock(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self, max_time: int, hidden_dim: int=256, embed_dim: int = 128) -> None:
        super().__init__()

        position = torch.arange(max_time, dtype=torch.float32, device=self.device).unsqueeze(1)  # (max_time, 1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=self.device) * -(
                math.log(1.0e4) / hidden_dim))  # (feature_dim / 2)
        self.pos_enc = torch.zeros((max_time, hidden_dim), dtype=torch.float32,
                                   device=self.device)  # (max_time, feature_dim)
        self.pos_enc[:, 0::2] = torch.sin(position * div_term)
        self.pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.time_embed_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Embedder for categorical attributes
        self.cell_index_embedder = nn.Embedding(256, hidden_dim)  # 256 cells in lon & lat ->
        self.day_embedder = nn.Embedding(33, hidden_dim)  # 31 days a month
        self.categorical_fc = nn.Linear(5 * hidden_dim, 5 * hidden_dim)


        # Embedder for numerical attributes
        self.numerical_fc = nn.Linear(3, hidden_dim)  # 3 for depart_time, traj_length, avg_move_distance

        self.mix_fc = nn.Sequential(
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(7 * hidden_dim, 7 * hidden_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(7 * hidden_dim, 7 * embed_dim),
            Swish(),
        )


    def forward(self, time: torch.Tensor, cat_attr: torch.Tensor, num_attr: torch.Tensor) -> torch.Tensor:
        """

        :param time: (B, )
        :param cat_attr: (B, 5) [start_lon, start_lat, end_lon, end_lat, day]
        :param num_attr: (B, 3) [depart_time, traj_length, avg_move_distance]
        :return:
        """
        time_embed = self.time_embed_layers(self.pos_enc[time, :])  # (B, embed_dim)

        cell_embed = self.cell_index_embedder(cat_attr[:, :-1]).flatten(1)  # (B, 4*embed_dim)
        day_embed = self.day_embedder(cat_attr[:, -1]) # (B, embed_dim)
        cat_embed = torch.cat([cell_embed, day_embed], dim=1) # (B, 5*embed_dim)

        num_embed = self.numerical_fc(num_attr)  # (B, embed_dim)

        overall_embed = torch.cat([time_embed, cat_embed, num_embed], dim=1)  # (B, 7*embed_dim)

        return self.mix_fc(overall_embed)  # (B, 7*embed_dim)


class MultiheadConvAttention(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_heads: int, c_reduce:int=1, dropout: float = 0.1) -> None:
        super().__init__()
        # input shape: (B, C, L)
        # we treat each channel as a token, so we have C tokens with L length
        self.num_heads = num_heads

        # Why compute q, k, v using Conv1d instead of linear?
        # In out setting, each token is a feature, the task is trajectory-related
        # so the elements in each token follow specific pattern or constraint
        # that is, the elements that are close to each other are more related
        # it is strange to update an element using all other elements in a token
        mid_c = in_c // c_reduce
        self.q_proj = nn.Conv1d(in_c, num_heads * in_c, 3, 1, 1, groups=in_c)
        self.k_proj = nn.Conv1d(in_c, num_heads * mid_c, 3, 1, 1, groups=mid_c)
        self.v_proj = nn.Conv1d(in_c, num_heads * mid_c, 3, 1, 1, groups=mid_c)

        self.attn_act = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        # This projection does not only merge all heads, but it is also a channel mixture
        # note that the previous operations are intra-channel operations (Inverse Transformer is inter-channel)
        # so this brings inter-channel communication
        self.out_proj = nn.Conv1d(in_c * num_heads, out_c, 1, 1, 0)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        if k is None:
            k = q
        if v is None:
            v = q
        # q, k, v: (B, C, L)
        q = self.q_proj(q).unflatten(1, (self.num_heads, -1))  # (B, H, C, L)
        kt = self.k_proj(k).unflatten(1, (self.num_heads, -1)).transpose(-1, -2)  # (B, H, L, C)
        v = self.v_proj(v).unflatten(1, (self.num_heads, -1))  # (B, H, C, L)

        attn = self.attn_act(torch.einsum('bhql,bhlk->bhqk', q, kt) / math.sqrt(q.shape[-1]))  # (B, H, C, C)

        return self.out_proj(torch.einsum('bhqk,bhkl->bhql', attn, v).flatten(1, 2))  # (B, C, L)


class ConvInvTransRes(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int, c_reduce:int=2, dropout: float = 0.1):
        super().__init__()

        self.stage1 = nn.Sequential(
            Swish(),
            MultiheadConvAttention(embed_dim, out_dim, num_heads, c_reduce, dropout),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(7 * 128, 7 * out_dim),
            Swish(),
            nn.Unflatten(1, (7, -1)), # (B, 7, embed_dim)
        )

        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)

        self.stage2 = nn.Sequential(
            Swish(),
            MultiheadConvAttention(out_dim, out_dim, num_heads, c_reduce, dropout),
        )

        self.shortcut = nn.Identity() if embed_dim == out_dim else nn.Conv1d(embed_dim, out_dim, 1,
                                                                             padding=0)  # (B, out_c, L)


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dim, L)
        # cond: (B, embed_dim*N)
        identity = self.shortcut(x)  # (B, out_dim, L)

        cond = self.cond_proj(cond)  # (B, N, out_dim)

        x = self.stage1(x).transpose(1, 2)  # (B, L, out_dim)

        x = self.multihead_attn(x, cond, cond)[0] + x  # (B, L, out_dim)

        x = x.transpose(1, 2)  # (B, out_dim, L)
        x = self.stage2(x)  # (B, out_dim, L)

        return identity + x  # (B, out_dim, L)


class TransRes(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            Swish(),
            nn.Conv1d(embed_dim, out_dim, 3, padding=1),  # (B, in_c, L)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(7 * 128, 7 * out_dim),
            Swish(),
            nn.Unflatten(1, (7, -1)), # (B, 7, embed_dim)
        )

        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout/2, batch_first=True)

        self.swish = Swish()
        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout/2, batch_first=True)

        self.shortcut = nn.Identity() if embed_dim == out_dim else nn.Conv1d(embed_dim, out_dim, 1, padding=0)  # (B, out_c, L)


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dim, L)
        # cond: (B, embed_dim*N)
        identity = self.shortcut(x).transpose(1, 2)  # (B, L, out_dim)
        cond = self.cond_proj(cond)  # (B, N, out_dim)

        x = self.stage1(x).transpose(1, 2)  # (B, L, out_dim)

        x = self.swish(self.cross_attn(x, cond, cond)[0] + x)  # (B, L, out_dim)

        return (identity + self.multihead_attn(x, x, x)[0]).transpose(1, 2)  # (B, out_dim, L)



if __name__ == '__main__':
    x = torch.randn(2, 128, 32)
    cond = torch.randn(2, 128*7)

    TR = TransRes(embed_dim=128, out_dim=128, num_heads=4)

    print(TR(x, cond).shape)

    # collect and free cache
    import gc
    gc.collect()
