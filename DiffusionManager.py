import torch
from typing import *
# import cv2

class DiffusionManager:
    def __init__(self, min_beta: float = 0.0001, max_beta: float = 0.002, max_diffusion_step: int = 100, device: str = 'cuda', scaled_linear:bool=False):
        if scaled_linear:
            betas = torch.linspace(min_beta**0.5, max_beta**0.5, max_diffusion_step).to(device) ** 2
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step
        self.betas = betas.view(-1, 1, 1)  # (T, 1, 1)
        self.alphas = alphas.view(-1, 1, 1)    # (T, 1, 1)
        self.alpha_bars = alpha_bars.view(-1, 1, 1)    # (T, 1, 1)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).view(-1, 1, 1)     # (T, 1, 1)
        self.sqrt_1_minus_alphas_bars = torch.sqrt(1 - alpha_bars).view(-1, 1, 1)    # (T, 1, 1)


    def diffusionForward(self, x_0, t, epsilon):
        """
        Forward Diffusion Process
        :param x_0: input (B, C, L)
        :param t: time steps (B, )
        :param epsilon: noise (B, C, L)
        :return: x_t: output (B, C, L)
        """
        x_t = self.sqrt_alpha_bars[t] * x_0 + self.sqrt_1_minus_alphas_bars[t] * epsilon
        return x_t


    def diffusionBackwardStep(self, x_t: torch.Tensor, t: int, epsilon_pred: torch.Tensor):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param epsilon_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        sqrt_1_minus_alpha_bar = self.sqrt_1_minus_alphas_bars[t].view(-1, 1, 1)

        mu = (x_t - beta / sqrt_1_minus_alpha_bar * epsilon_pred) / torch.sqrt(alpha)
        if t == 0:
            return mu
        else:
            stds = torch.sqrt(beta) * torch.randn_like(x_t)
            return mu + stds

    @torch.no_grad()
    def diffusionBackward(self, x_T: torch.Tensor, cat_attr: torch.Tensor, num_attr: torch.Tensor, model: torch.nn.Module, max_t: int = None):
        """
        Backward Diffusion Process
        :param x_T: input (B, C, L)
        :param model: model to predict noise
        :param max_t: maximum time step
        :return: x_0: output (B, C, L)
        """
        if max_t is None:
            max_t = self.T
        B = x_T.shape[0]
        x_t = x_T
        tensor_t = torch.arange(max_t, dtype=torch.long, device=x_t.device).repeat(B, 1)  # (B, T)
        for t in range(max_t - 1, -1, -1):
            epsilon_pred = model(x_t, tensor_t[:, t], cat_attr, num_attr)  # epsilon_pred: (B, C, L)
            x_t = self.diffusionBackwardStep(x_t, t, epsilon_pred)
            # if t % 10 == 0:
            #     x_np = x_t.squeeze().cpu().numpy()
            #     print(f"min={x_np.min()}, max={x_np.max()}")
            #     cv2.imshow('x_t', cv2.resize(x_np, (512, 512)))
            #     cv2.waitKey(0)
        return x_t


