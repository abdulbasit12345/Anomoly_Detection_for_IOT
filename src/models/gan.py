import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """
    Generator with residual skip connection.
    Same architecture as before — tanh output keeps features in [-1, 1] range.
    """

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.Tanh(),
        )

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, output_dim),
        )

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.main(z) + self.proj(z))


class Critic(nn.Module):
    """
    WGAN-GP Critic (replaces the old Discriminator).

    KEY DIFFERENCE from the old Discriminator:
    - NO Sigmoid at the end → outputs an unbounded real-valued score.
    - This is mathematically required by Wasserstein distance.
    - The old Sigmoid caused the discriminator to "saturate" (output exactly 0 or 1)
      which killed generator gradients and caused mode collapse.
    - Spectral normalisation is REMOVED (it conflicts with gradient penalty).
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
            # No Sigmoid — WGAN critic outputs a raw score
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def compute_gradient_penalty(critic: Critic, real: torch.Tensor,
                              fake: torch.Tensor, device: torch.device,
                              lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Gradient Penalty (WGAN-GP): enforces the 1-Lipschitz constraint on the critic.

    Interpolates between real and fake samples, computes critic output,
    then penalises gradients whose L2 norm deviates from 1.
    This replaces weight clipping (original WGAN) for more stable training.
    """
    batch_size = real.size(0)
    # Random interpolation weight α ∈ [0, 1]
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)

    interpolated = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)
    critic_interp = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradients and compute L2 norm per sample
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gp


def build_gan(config, n_features: int, device: torch.device):
    """Build and return (Generator, Critic) on the specified device."""
    G = Generator(config.LATENT_DIM, n_features).to(device)
    C = Critic(n_features).to(device)
    return G, C