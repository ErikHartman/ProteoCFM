import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, Optional


class BaseNetwork(nn.Module):
    # Could technically remove but seeded run so keep for reproducibility
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        covariates: Optional[Dict[str, Any]] = None,
        covariate_embed_dim: int = 16,
        time_embed_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.hidden_dim = hidden_dim
        self.covariate_embed_dim = covariate_embed_dim
        self.time_embed_dim = time_embed_dim
        self.n_blocks = n_blocks
        self.dropout = dropout

        self.net = ResidualBase(
            input_dim,
            hidden_dim,
            self.output_dim,
            n_blocks,
            dropout,
            time_embed_dim,
            covariates,
            covariate_embed_dim,
        )

    def _init_embeddings(self, covariate_info: Dict[str, Any]):
        new_cov_embedding = CovariateEmbedding(covariate_info, self.covariate_embed_dim)
        self.net.cov_embedding = new_cov_embedding
        
        new_total_dim = self.time_embed_dim + new_cov_embedding.total_dim
        self.net._cond_dim = new_total_dim
        self.net.film = FiLM(new_total_dim, self.hidden_dim)
        for block in self.net.blocks:
            block.film = FiLM(new_total_dim, self.hidden_dim)

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        covariates: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        return self.net.forward(x, time, covariates)


class ResidualBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_blocks: int = 3,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        covariates: Optional[Dict[str, Any]] = None,
        covariate_embed_dim: int = 16,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.cov_embedding = CovariateEmbedding(
            covariates, embed_dim=covariate_embed_dim
        )
        self._cond_dim = time_embed_dim + self.cov_embedding.total_dim
        self.film = FiLM(self._cond_dim, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout, cond_dim=self._cond_dim) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self, x: Tensor, time: Tensor, covariates: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        cond_parts = []
        t_emb = self.time_embedding(time)
        cond_parts.append(t_emb)
        cov_emb = self.cov_embedding(covariates)
        if cov_emb is not None:
            cond_parts.append(cov_emb)
        cond = torch.cat(cond_parts, dim=-1)
        
        h = self.input_proj(x)
        h = self.film(h, cond)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(max_period))
            * torch.arange(half_dim)
            / max(half_dim - 1, 1)
        )
        self.register_buffer("freq", freq, persistent=True)

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        t = t.to(dtype=self.freq.dtype)
        args = t[:, None] * self.freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat(
                [emb, emb.new_zeros(emb.shape[0], self.dim - emb.shape[-1])], dim=-1
            )
        return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
        )

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return (1 + gamma) * h + beta


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, cond_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.film = FiLM(cond_dim, dim) if cond_dim > 0 else None

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        h = self.net(x)
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        return x + h


class CovariateEmbedding(nn.Module):
    def __init__(
        self, covariates: Optional[Dict[str, Any]] = None, embed_dim: int = 16
    ):
        super().__init__()
        if covariates is None:
            covariates = {}

        if "categorical" in covariates or "continuous" in covariates:
            categorical_covariates = covariates.get("categorical", {}) or {}
            continuous_covariates = covariates.get("continuous", []) or []
        else:
            categorical_covariates = covariates
            continuous_covariates = []

        self.categorical_names = list(categorical_covariates.keys())
        self.continuous_names = list(continuous_covariates)
        self.covariate_names = self.categorical_names + self.continuous_names
        self.covariate_embeds = nn.ModuleDict()
        self.continuous_proj = None

        self.total_dim = 0
        for name, n_categories in categorical_covariates.items():
            self.covariate_embeds[name] = nn.Embedding(n_categories, embed_dim)
            self.total_dim += embed_dim

        if self.continuous_names:
            self.continuous_proj = nn.Sequential(
                nn.Linear(len(self.continuous_names), embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.total_dim += embed_dim

    def forward(
        self, covariates: Optional[Dict[str, Tensor]] = None
    ) -> Optional[Tensor]:
        """Returns concatenated covariate embeddings, or None if no covariates."""
        if not self.covariate_names:
            return None

        if covariates is None:
            covariates = {}

        emb_parts = []
        for name in self.categorical_names:
            if name not in covariates:
                raise ValueError(f"Covariate '{name}' is required but not provided")
            cov_emb = self.covariate_embeds[name](covariates[name])
            emb_parts.append(cov_emb)

        if self.continuous_names:
            continuous_values = []
            for name in self.continuous_names:
                if name not in covariates:
                    raise ValueError(f"Covariate '{name}' is required but not provided")
                value = covariates[name].float()
                if value.dim() > 1:
                    value = value.squeeze(-1)
                continuous_values.append(value)
            continuous_tensor = torch.stack(continuous_values, dim=-1)
            emb_parts.append(self.continuous_proj(continuous_tensor))

        return torch.cat(emb_parts, dim=-1)
