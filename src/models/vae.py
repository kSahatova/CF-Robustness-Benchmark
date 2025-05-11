# pyright: reportMissingImports=false

import math
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(self, input_channels, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=4, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )  # 7x7 -> 4x4
        self.fc_mu = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(
            32, output_channels, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 32x32
        self.output_layer = nn.Conv2d(
            output_channels, output_channels, kernel_size=5, stride=1, padding=0
        )  # 32x32 -> 28x28

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, 128, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        x_recon = torch.sigmoid(self.output_layer(h))
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_channels, z_dim, beta):
        super(VAE, self).__init__()
        self.beta = beta
        self.encoder = Encoder(input_channels, z_dim)
        self.decoder = Decoder(z_dim, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(
        self, total_steps, shape="linear", baseline=0.0, cyclical=False
    ):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
        """

        self.current_step = 0

        if shape not in ["linear", "cosine", "logistic"]:
            raise ValueError("Shape must be one of 'linear', 'cosine', or 'logistic.")
        self.shape = shape

        if not 0 <= float(baseline) <= 1:
            raise ValueError("Baseline must be a float between 0 and 1.")
        self.baseline = baseline

        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("Argument total_steps must be an integer greater than 0")
        self.total_steps = total_steps

        if type(cyclical) is not bool:
            raise ValueError("Argument cyclical must be a boolean.")
        self.cyclical = cyclical


    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        self.weight = self._slope()
        out = kld * self.weight
        return out

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def _slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1.0
        y = self._add_baseline(y)
        return y

    def _add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out


class BetaVAE(nn.Module):
    """Another implementation of beta-VAE with a more modular architecture"""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128, 256, 512],
        latent_dim: int = 128,
        input_size: Tuple[int, int] = (224, 224),
        beta: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_size = input_size

        # Calculate the size of the flattened feature map
        self.feature_size = (
            self.input_size[0] // (2 ** len(self.hidden_dims)),
            self.input_size[1] // (2 ** len(self.hidden_dims)),
        )
        self.flattened_dim = (
            self.hidden_dims[-1] * self.feature_size[0] * self.feature_size[1]
        )

        # Build Encoder
        modules = []
        in_channels = self.input_channels

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder_ = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.flattened_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.flattened_dim, self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, self.flattened_dim)

        hidden_dims_reversed = self.hidden_dims[::-1]

        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear"),
                    nn.Conv2d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                    nn.ReLU(),
                )
            )

        self.decoder_ = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                hidden_dims_reversed[-1],
                kernel_size=7,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.BatchNorm2d(hidden_dims_reversed[-1]),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dims_reversed[-1], self.input_channels, kernel_size=2, padding=0
            ),
            nn.Sigmoid(),
        )

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder_(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(
            -1, self.hidden_dims[-1], self.feature_size[0], self.feature_size[1]
        )
        result = self.decoder_(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var
