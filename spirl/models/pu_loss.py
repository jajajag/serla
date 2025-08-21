# spirl/models/pu_loss.py
import torch

def pu_risk(dz_pos: torch.Tensor,
            dz_all: torch.Tensor,
            lambda_p: float = 0.6,
            xi: float = 0.2) -> torch.Tensor:
    """
    dz_pos: D(z) for positive (expert) samples, shape (B,)
    dz_all: D(z) for unlabeled samples, shape (B,)
    Returns nnPU risk to MINIMIZE (for the discriminator). For generator side, use -rho * pu_risk.
    """
    eps = 1e-6
    dz_pos = dz_pos.clamp_min(eps).clamp_max(1 - eps)
    dz_all = dz_all.clamp_min(eps).clamp_max(1 - eps)

    # positive risk (as positive class)
    L1_pos = (-torch.log(dz_pos)).mean()
    # negative risks
    L0_all = (-torch.log(1.0 - dz_all)).mean()
    L0_pos = (-torch.log(1.0 - dz_pos)).mean()

    rhs = L0_all - lambda_p * L0_pos
    return lambda_p * L1_pos + torch.maximum(-torch.as_tensor(xi, device=dz_pos.device), rhs)

