import torch
from typing import Callable, Tuple

def add_skill_noise(z: torch.Tensor, sigma: float = 1e-2) -> torch.Tensor:
    return z + sigma * torch.randn_like(z)

def recon_loss_with_noise(encoder, decoder, a_seq: torch.Tensor,
                          sigma: float, loss_fn: Callable) -> torch.Tensor:
    # encoder should return (z, *extras)
    z, *extras = encoder(a_seq)
    a_hat = decoder(add_skill_noise(z, sigma))
    return loss_fn(a_hat, a_seq)

