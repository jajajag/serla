import torch
import torch.nn as nn
from typing import Sequence

class PUDiscriminator(nn.Module):
    """MLP Discriminator D_ζ(z) -> (0,1)."""

    def __init__(self, latent_dim: int,
                 hidden_sizes: Sequence[int] = (256, 256),
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)  # logit
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        h = self.backbone(z)
        logits = self.head(h).squeeze(-1)          # (B,)
        if return_logits:
            return logits
        return torch.sigmoid(logits)               # (0,1)

