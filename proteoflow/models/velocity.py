from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from .modules import BaseNetwork


class VelocityNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        covariates: Optional[Dict[str, Any]] = None,
        covariate_embed_dim: int = 16,
        time_embed_dim: int = 128,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.covariate_embed_dim = covariate_embed_dim
        self.time_embed_dim = time_embed_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.architecture = "single"

        self.net = BaseNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            covariates=covariates,
            covariate_embed_dim=covariate_embed_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )

    def _init_embeddings(self, covariate_info: Dict[str, Any]):
        self.net._init_embeddings(covariate_info)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        return self.net.forward(x, time=t, covariates=covariates)

    def forward_components(
        self,
        x: Tensor,
        t: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        total = self.forward(x, t, covariates)
        return {"total": total}

    @torch.no_grad()
    def integrate_single(
        self,
        x0: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        n_steps: int = 50,
    ):
        dt = (t_end - t_start) / n_steps
        x = x0
        trajectory = [x0]
        for _ in range(n_steps):
            t_cur_tensor = torch.tensor([t_start], dtype=x.dtype, device=x.device)
            t_mid_tensor = torch.tensor([t_start + dt / 2], dtype=x.dtype, device=x.device)
            k1 = self.forward(x, t_cur_tensor, covariates)
            x_mid = x + k1 * (dt / 2)
            k2 = self.forward(x_mid, t_mid_tensor, covariates)
            x = x + k2 * dt
       
            trajectory.append(x)
            t_start += dt
        return trajectory

    @torch.no_grad()
    def integrate(
        self,
        x0: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
        t_start: Tensor = None,
        t_end: Tensor = None,
        n_steps: int = 50,
    ):
        batch_size = x0.shape[0]

        if isinstance(t_start, (int, float)):
            t_start = torch.full((batch_size,), t_start, device=x0.device, dtype=x0.dtype)
        if isinstance(t_end, (int, float)):
            t_end = torch.full((batch_size,), t_end, device=x0.device, dtype=x0.dtype)

        trajectories = []
        for i in range(batch_size):
            sample_covariates = (
                {k: v[i : i + 1] for k, v in covariates.items()} if covariates else None
            )
            traj = self.integrate_single(
                x0=x0[i : i + 1],
                covariates=sample_covariates,
                t_start=t_start[i].item(),
                t_end=t_end[i].item(),
                n_steps=n_steps,
            )
            trajectories.append(torch.stack(traj, dim=1))

        return torch.cat(trajectories, dim=0)
