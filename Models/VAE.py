from Models.VAEBlocks import *


class BaseVAE(nn.Module):
    def __init__(self, no_noise_ratio: float = 0):
        super().__init__()
        self.noise_ratio = 1 - no_noise_ratio
        self.encoder = nn.Sequential(
            ConvNormAct(2, 128, 3, 1, 1),  # (B, 128, L)
            nn.AvgPool1d(8, 8),  # (B, 128, L/8)
        )

        self.mu_head = nn.Conv1d(128, 4, 1, 1, 0)  # (B, 4, L/8), n_ele = 0.5L
        self.logvar_head = nn.Conv1d(128, 4, 1, 1, 0)  # (B, 4, L/8), n_ele = 0.5L

        self.decoder = nn.Sequential(
            ConvNormAct(4, 128, 3, 1, 1),  # (B, 128, L/8)
            nn.Upsample(scale_factor=8, mode='linear', align_corners=True),  # (B, 128, L)
            nn.Conv1d(128, 2, 1, 1, 0)  # (B, 2, L)
        )


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        noise_mask = torch.rand_like(mu) < self.noise_ratio
        std = torch.exp(0.5 * logvar) * noise_mask
        eps = torch.randn_like(std)
        return eps * std + mu  # (B, L/2), n_ele = L/2


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc_x = self.encoder(x)
        return self.mu_head(enc_x)


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        :param x: input tensor, (B, 2, L), n_ele = 2L
        :return:
                    recovered input x: (B, 2, L), n_ele = 2L
                    mu: (B, 4, L/8), n_ele = 0.5L
                    logvar: (B, 4, L/8), n_ele = 0.5L
        """

        enc_x = self.encoder(x)  # (B, 32, L/8), n_ele = 4L
        mu, logvar = self.mu_head(enc_x), self.logvar_head(enc_x)  # (B, 4, L/8), (B, 4, L/8)

        z = self.reparameterize(mu, logvar)  # (B, 4, L/8), n_ele = L/2

        recover_x = self.decoder(z)  # (B, 2, L), n_ele = 4L

        return recover_x, mu, logvar


# speed: 100 it/s
class VAE_SERes(BaseVAE):
    def __init__(self, no_noise_ratio: float = 0) -> None:
        super().__init__(no_noise_ratio)
        # input: (B, 2, L), n_ele = 2L
        # target: (B, L), n_ele = 0.5L

        self.encoder = nn.Sequential(
            ConvNormAct(2, 16, 3, 1, 1),  # (B, 16, L)

            ConvNormAct(16, 32, 3, 2, 1),   # (B, 32, L/2)
            SEResBlock(32),   # (B, 32, L/2)

            ConvNormAct(32, 64, 3, 2, 1),   # (B, 64, L/4)
            SEResBlock(64),   # (B, 64, L/4)

            ConvNormAct(64, 128, 3, 2, 1),   # (B, 128, L/8)
            SEResBlock(128),   # (B, 128, L/8)

            AttentionBlock(embed_dim=128, num_heads=4, dropout=0.1),   # (B, 128, L/8)
        )

        self.decoder = nn.Sequential(
            ConvNormAct(4, 128, 3, 1, 1),  # (B, 128, L/8), n_ele = 4L
            AttentionBlock(embed_dim=128, num_heads=4, dropout=0.1),  # (B, 4, L/8)

            SEResBlock(128),  # (B, 128, L/8)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 128, L/4), n_ele = 32L
            ConvNormAct(128, 64, 3, 1, 1),  # (B, 64, L/4), n_ele = 16L

            SEResBlock(64),  # (B, 64, L/4)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 64, L/2), n_ele = 32L
            ConvNormAct(64, 32, 3, 1, 1),  # (B, 32, L/2), n_ele = 16L

            SEResBlock(32),  # (B, 32, L/2)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 32, L), n_ele = 32L
            ConvNormAct(32, 16, 3, 1, 1),  # (B, 16, L), n_ele = 16L

            nn.Conv1d(16, 2, 3, 1, 1),  # (B, 2, L), n_ele = L
        )


# Total params: 72,586
# MADDs: 38.00M
# speed: 242 - 307 it/s
class VAE_PDR(BaseVAE):
    def __init__(self, no_noise_ratio: float) -> None:
        super().__init__(no_noise_ratio)
        # input: (B, 2, L), n_ele = 2L
        # target: (B, L), n_ele = 0.5L

        self.encoder = nn.Sequential(
            ConvNormAct(2, 16, 3, 1, 1),     # (B, 16, L), n_ele = 16L
            nn.AvgPool1d(3, 2, 1),     # (B, 32, L/2), n_ele = 8L
            ConvNormAct(16, 32, 3, 1, 1),  # (B, 32, L/2), n_ele = 16L
            nn.AvgPool1d(3, 2, 1),     # (B, 128, L/4), n_ele = 8L
            ParallelDilationRes(32),     # (B, 32, L/4), n_ele = 8L     # extend receptive field with multi
            ConvNormAct(32, 64, 3, 2, 1),  # (B, 32, L/8), n_ele = 4L
            ParallelDilationRes(64)
        )

        self.mu_head = nn.Conv1d(64, 4, 3, 1, 1)  # (B, 4, L/8), n_ele = 0.5L
        self.logvar_head = nn.Conv1d(64, 4, 3, 1, 1)    # (B, 4, L/8), n_ele = 0.5L

        self.decoder = nn.Sequential(
            ConvNormAct(4, 32, 3, 1, 1),  # (B, 32, L/8), n_ele = 4L
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 32, L/4), n_ele = 8L
            ConvNormAct(32, 64, 3, 1, 1),  # (B, 64, L/4), n_ele = 16L
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 64, L/2), n_ele = 32L
            ParallelDilationRes(64),  # (B, 64, L/2), n_ele = 64L
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 64, L), n_ele = 64L
            nn.Conv1d(64, 2, 3, 1, 1),  # (B, 2, L), n_ele = L
        )



class VAE_InvTrans(BaseVAE):
    def __init__(self, no_noise_ratio: float = 0) -> None:
        super().__init__(no_noise_ratio)
        # input: (B, 2, L), n_ele = 2L
        # target: (B, L), n_ele = 0.5L

        self.encoder = nn.Sequential(
            nn.Upsample(size=256, mode='linear', align_corners=True),  # (B, 2, 256)
            ConvNormAct(2, 16, 3, 1, 1),  # (B, 16, 256)

            ConvNormAct(16, 32, 3, 2, 1),   # (B, 32, 128)
            InvTransformerBlock(in_c=32, in_seq_len=128, num_heads=8),   # (B, 32, 128)

            ConvNormAct(32, 64, 3, 2, 1),   # (B, 64, 64)
            InvTransformerBlock(in_c=64, in_seq_len=64, num_heads=8),   # (B, 64, 64)

            ConvNormAct(64, 128, 3, 2, 1),   # (B, 128, 32)
            InvTransformerBlock(in_c=128, in_seq_len=32, num_heads=8),   # (B, 128, 32)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(size=32, mode='linear', align_corners=True),  # (B, 128, 32)
            ConvNormAct(4, 128, 3, 1, 1),  # (B, 128, 32)

            InvTransformerBlock(in_c=128, in_seq_len=32, num_heads=8),   # (B, 128, 32)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 128, 64)
            ConvNormAct(128, 64, 3, 1, 1),  # (B, 64, 64)

            InvTransformerBlock(in_c=64, in_seq_len=64, num_heads=8),   # (B, 64, 64)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 64, 128)
            ConvNormAct(64, 32, 3, 1, 1),  # (B, 32, 128)

            InvTransformerBlock(in_c=32, in_seq_len=128, num_heads=8),   # (B, 32, 128)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # (B, 32, 256)
            ConvNormAct(32, 16, 3, 1, 1),  # (B, 16, 256)

            nn.Conv1d(16, 2, 3, 1, 1),  # (B, 2, L), n_ele = L
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input x into latent space.

        :param x: input tensor, (B, 2, L), n_ele = 2L
        :return: latent tensor, (B, 128, L/8), n_ele = 16L
        """
        _, _, L = x.shape
        x = F.adaptive_avg_pool1d(self.encoder(x), L//8)  # (B, 128, L/8)
        return self.mu_head(x)


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor into output x.

        :param z: latent tensor, (B, 128, L/8), n_ele = 16L
        :return: output tensor, (B, 2, L), n_ele = 2L
        """
        _, _, L = z.shape
        return F.adaptive_avg_pool1d(self.decoder(z), L)  # (B, 2, L), n_ele = 2L


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        :param x: input tensor, (B, 2, L), n_ele = 2L
        :return:
                    recovered input x: (B, 2, L), n_ele = 2L
                    mu: (B, 4, L/8), n_ele = 0.5L
                    logvar: (B, 4, L/8), n_ele = 0.5L
        """

        B, C, L = x.shape

        enc_x = self.encoder(x)  # (B, 128, L/8)
        enc_x = F.adaptive_avg_pool1d(enc_x, L//8) # (B, 128, L/8)
        mu, logvar = self.mu_head(enc_x), self.logvar_head(enc_x)  # (B, 4, L/8), (B, 4, L/8)

        z = self.reparameterize(mu, logvar)  # (B, 4, L/8), n_ele = L/2

        recover_x = self.decoder(z)  # (B, 2, L), n_ele = 4L
        recover_x = F.adaptive_avg_pool1d(recover_x, L)  # (B, 2, L)

        return recover_x, mu, logvar


class VAELoss(nn.Module):
    def __init__(self, kld_coef: float) -> None:
        super().__init__()
        self.kld_coef = kld_coef

    def forward(self, traj: torch.Tensor, recover_traj: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # recover: (B, 4, L), n_ele = 4L
        # target: (B, 4, L), n_ele = 4L
        # mu: (B, L/4), logvar: (B, L/4)
        traj_length = traj.shape[-1]
        start_mse = F.mse_loss(recover_traj[:, :, 0], traj[:, :, 0], reduction="sum") # the first point
        end_mse = F.mse_loss(recover_traj[:, :, -1], traj[:, :, -1], reduction="sum") # the last point
        traj_mse = F.mse_loss(recover_traj, traj, reduction="sum") / traj_length
        mse = start_mse + end_mse + traj_mse

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / traj_length
        return mse + self.kld_coef * kld, mse.item(), self.kld_coef * kld.item()


if __name__ == '__main__':
    # from torchsummary import summary
    # model = VAE()
    # summary(model, torch.randn(1, 2, 1000))

    model = VAE_InvTrans(0.1).cuda()

    print(model.encode(torch.randn(1, 2, 199).cuda()).shape)

    # for i in range(16, 256):
    #     x = torch.randn(1, 2, i).cuda()
    #     recover_x, mu, logvar = model(x)
    #     print(recover_x.shape, mu.shape, logvar.shape)
    # for i in range(10):
    #     inferSpeedTest1K(model, torch.randn(1, 2, 1000).cuda())