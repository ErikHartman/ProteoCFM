from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .velocity import VelocityNetwork
from ..training.callbacks import EarlyStopping


def _empty_trajectory_result(
    feature_cols: List[str],
    covariate_cols: List[str],
    protein_cols: Optional[List[str]] = None,
) -> dict:
    meta = ["path_id", "start_idx", "rep", "step", "t"] + covariate_cols
    result = {
        "pca_abundances": pd.DataFrame(columns=meta + feature_cols),
        "pca_velocities": pd.DataFrame(columns=meta + ["velocity"]),
        "protein_abundances": None,
        "protein_velocities": None,
    }
    if protein_cols is not None:
        result["protein_abundances"] = pd.DataFrame(columns=meta + protein_cols)
        result["protein_velocities"] = pd.DataFrame(columns=meta + protein_cols)
    return result


def _encode_covariate(data_module, col: str, raw_val: Any) -> int:
    try:
        return data_module.covariate_categories[col].index(raw_val)
    except ValueError as exc:
        raise ValueError(
            f"Unknown or missing covariate value {raw_val!r} for '{col}'."
        ) from exc


def _normalize_continuous_covariate(data_module, col: str, raw_val: Any) -> float:
    stats = data_module.continuous_covariate_stats.get(col, {"mean": 0.0, "std": 1.0})
    return (float(raw_val) - stats["mean"]) / stats["std"]


def sample_trajectories(
    data_module,
    flow_model,
    split: str = "train",
    filter_by: Optional[Dict[str, Any]] = None,
    t_start_max: Optional[float] = None,
    t_end: float = 1.0,
    n_steps: int = 50,
    n_trajectories_per_start: int = 1,
    pca_reducer=None,
    protein_cols: Optional[List[str]] = None,
) -> dict:
    df = data_module.df.copy()
    feature_cols = data_module.feature_cols
    categorical_covariate_cols = data_module.covariate_cols or []
    continuous_covariate_cols = getattr(data_module, "continuous_covariate_cols", []) or []
    covariate_cols = categorical_covariate_cols + continuous_covariate_cols
    device = next(flow_model.velocity_net.parameters()).device
    flow_model.velocity_net.eval()

    if split != "all":
        df = df[df["split"] == split]

    if filter_by:
        for col, val in filter_by.items():
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    if t_start_max is not None:
        df = df[df["time"] <= t_start_max]

    df = df.reset_index(drop=True)

    rows = []
    path_id = 0

    for start_i, row_meta in df.iterrows():
        x0_np = row_meta[feature_cols].values.astype(np.float32)
        t_start = float(row_meta["time"])

        covariates = None
        cov_values = {}
        if covariate_cols:
            covariates = {}
            for col in categorical_covariate_cols:
                raw_val = row_meta[col]
                cov_values[col] = raw_val
                code = _encode_covariate(data_module, col, raw_val)
                covariates[col] = torch.tensor([code], dtype=torch.long, device=device)
            for col in continuous_covariate_cols:
                raw_val = row_meta[col]
                cov_values[col] = raw_val
                norm_val = _normalize_continuous_covariate(data_module, col, raw_val)
                covariates[col] = torch.tensor([norm_val], dtype=torch.float32, device=device)

        for rep in range(n_trajectories_per_start):
            x0 = torch.tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                traj = flow_model.integrate(
                    x0=x0,
                    covariates=covariates,
                    t_start=t_start,
                    t_end=t_end,
                    n_steps=n_steps,
                )

            traj_tensor = traj[0]
            traj_np = traj_tensor.cpu().numpy()
            t_vals = np.linspace(t_start, t_end, traj_np.shape[0])

            t_batch = torch.tensor(t_vals, dtype=torch.float32, device=device)
            cov_batch = (
                {name: value.expand(len(t_vals)) for name, value in covariates.items()}
                if covariates is not None
                else None
            )
            with torch.no_grad():
                v_pca_batch = flow_model.velocity_net(traj_tensor, t_batch, cov_batch)
            v_pca_np = v_pca_batch.cpu().numpy()
            v_magnitudes = np.linalg.norm(v_pca_np, axis=1)

            traj_protein = None
            v_prot_np = None
            if pca_reducer is not None and protein_cols is not None:
                traj_protein = pca_reducer.inverse_transform(traj_np)
                v_prot_np = pca_reducer.inverse_transform_velocity(v_pca_np)

            for step in range(traj_np.shape[0]):
                row = {
                    "path_id": path_id,
                    "start_idx": start_i,
                    "rep": rep,
                    "step": step,
                    "t": t_vals[step],
                    "velocity": float(v_magnitudes[step]),
                }
                row.update(cov_values)
                for fi, col in enumerate(feature_cols):
                    row[col] = traj_np[step, fi]
                if traj_protein is not None:
                    for pi, col in enumerate(protein_cols):
                        row[col] = traj_protein[step, pi]
                        row["v_" + col] = float(v_prot_np[step, pi])
                rows.append(row)

            path_id += 1

    trajectory_df = pd.DataFrame(rows)
    if trajectory_df.empty:
        return _empty_trajectory_result(feature_cols, covariate_cols, protein_cols)

    meta = [
        col
        for col in ["path_id", "start_idx", "rep", "step", "t", *covariate_cols]
        if col in trajectory_df.columns
    ]

    result = {
        "pca_abundances": trajectory_df[meta + feature_cols].copy(),
        "pca_velocities": trajectory_df[meta + ["velocity"]].copy(),
        "protein_abundances": None,
        "protein_velocities": None,
    }

    if pca_reducer is not None and protein_cols is not None:
        v_prot_cols = ["v_" + col for col in protein_cols]
        result["protein_abundances"] = trajectory_df[meta + protein_cols].copy()
        result["protein_velocities"] = trajectory_df[meta + v_prot_cols].rename(
            columns={"v_" + col: col for col in protein_cols}
        ).copy()

    return result


class FlowMatchingModel:
    def __init__(
        self,
        velocity_net: VelocityNetwork,
        n_t_samples: int = 3,
    ):
        self.velocity_net = velocity_net
        self.n_t_samples = n_t_samples
        self.model_type = getattr(velocity_net, "architecture", "single")

    @classmethod
    def single(
        cls,
        *,
        input_dim: int,
        covariate_embed_dim: int = 16,
        time_embed_dim: int = 128,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        n_t_samples: int = 3,
    ) -> "FlowMatchingModel":
        velocity_net = VelocityNetwork(
            input_dim=input_dim,
            covariate_embed_dim=covariate_embed_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
        )
        return cls(velocity_net=velocity_net, n_t_samples=n_t_samples)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        return self.velocity_net.forward(x, t, covariates)

    def forward_components(
        self,
        x: Tensor,
        t: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        return self.velocity_net.forward_components(x, t, covariates)

    def integrate(
        self,
        x0: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
        t_start: Optional[Tensor] = None,
        t_end: Optional[Tensor] = None,
        n_steps: int = 50,
    ) -> Tensor:
        return self.velocity_net.integrate(x0, covariates, t_start, t_end, n_steps)

    def integrate_single(
        self,
        x0: Tensor,
        covariates: Optional[Dict[str, Tensor]] = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        n_steps: int = 50,
    ):
        return self.velocity_net.integrate_single(x0, covariates, t_start, t_end, n_steps)

    def _build_covariate_info(
        self,
        data_module,
        covariates: Optional[List[str]],
        continuous_covariates: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        categorical = list(covariates or [])
        continuous = list(continuous_covariates or [])
        if not categorical and not continuous:
            return None

        return {
            "categorical": {
                cov_name: len(data_module.covariate_categories[cov_name])
                for cov_name in categorical
            },
            "continuous": continuous,
        }

    def _compute_loss_terms(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x0, x1, t_interp = batch["x0"], batch["x1"], batch["t_interp"]
        t0, t1 = batch["t0"], batch["t1"]
        covariates = batch.get("covariates")

        n_t = self.n_t_samples
        if n_t > 1:
            x0 = x0.repeat_interleave(n_t, dim=0)
            x1 = x1.repeat_interleave(n_t, dim=0)
            t0 = t0.repeat_interleave(n_t, dim=0)
            t1 = t1.repeat_interleave(n_t, dim=0)
            if covariates is not None:
                covariates = {k: v.repeat_interleave(n_t, dim=0) for k, v in covariates.items()}
            dt = (t1 - t0).clamp_min(1e-6)
            t_interp = t0 + torch.rand_like(t0) * dt
        else:
            dt = (t1 - t0).clamp_min(1e-6)

        alpha = ((t_interp - t0) / dt).unsqueeze(-1)
        x_t = (1 - alpha) * x0 + alpha * x1
        target_velocity = (x1 - x0) / dt.unsqueeze(-1)

        pred_velocity = self.velocity_net.forward(x_t, t_interp, covariates=covariates)
        fit_loss = ((pred_velocity - target_velocity) ** 2).sum(dim=-1).mean()

        return {
            "loss": fit_loss,
            "fit_loss": fit_loss,
        }

    def fit(
        self,
        data_module,
        covariates: Optional[List[str]] = None,
        continuous_covariates: Optional[List[str]] = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        lr_min_delta: float = 1e-4,
        patience: int = 50,
        min_delta: float = 1e-4,
        resample_pairs_each_epoch: bool = True,
    ) -> Tuple["FlowMatchingModel", Dict[str, Any]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if covariates is None:
            covariates = getattr(data_module, "covariate_cols", [])
        if continuous_covariates is None:
            continuous_covariates = getattr(data_module, "continuous_covariate_cols", [])

        covariate_info = self._build_covariate_info(data_module, covariates, continuous_covariates)
        if covariate_info is not None:
            self.velocity_net._init_embeddings(covariate_info)

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        has_validation = val_loader is not None

        self.velocity_net.to(device)
        opt = torch.optim.Adam(self.velocity_net.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=lr_min_delta,
        )

        early_stop = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            save_best_state=True,
        )

        train_loss_history = []
        val_loss_history = []
        train_fit_loss_history = []
        val_fit_loss_history = []
        effective_epochs = 0

        self.velocity_net.train()

        for epoch in range(n_epochs):
            if resample_pairs_each_epoch and epoch > 0:
                data_module.resample_train_pairs()
                train_loader = data_module.train_dataloader()
            epoch_train_stats = []

            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                opt.zero_grad(set_to_none=True)
                loss_terms = self._compute_loss_terms(batch)
                loss = loss_terms["loss"]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), max_norm=20.0)

                opt.step()
                epoch_train_stats.append({key: value.item() for key, value in loss_terms.items()})

            mean_train_loss = float(np.mean([stat["loss"] for stat in epoch_train_stats]))
            train_loss_history.append(mean_train_loss)
            train_fit_loss_history.append(float(np.mean([stat["fit_loss"] for stat in epoch_train_stats])))
            effective_epochs += 1

            if has_validation:
                self.velocity_net.eval()
                epoch_val_stats = []

                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        loss_terms = self._compute_loss_terms(batch)
                        epoch_val_stats.append({key: value.item() for key, value in loss_terms.items()})

                mean_val_loss = float(np.mean([stat["loss"] for stat in epoch_val_stats]))
                val_loss_history.append(mean_val_loss)
                val_fit_loss_history.append(float(np.mean([stat["fit_loss"] for stat in epoch_val_stats])))
                self.velocity_net.train()

                scheduler.step(mean_val_loss)

                if early_stop(mean_val_loss, self.velocity_net):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                scheduler.step(mean_train_loss)
                if early_stop(mean_train_loss, self.velocity_net):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if epoch % max(1, n_epochs // 10) == 0:
                if has_validation:
                    print(f"Epoch {epoch + 1}/{n_epochs} | Train: {mean_train_loss:.4f} | Val: {mean_val_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{n_epochs} | Train: {mean_train_loss:.4f}")

        self.velocity_net.eval()
        for p in self.velocity_net.parameters():
            p.requires_grad_(False)

        metrics: Dict[str, Any] = {
            "train_losses": train_loss_history,
            "val_losses": val_loss_history if has_validation else [],
            "train_fit_losses": train_fit_loss_history,
            "val_fit_losses": val_fit_loss_history if has_validation else [],
            "final_train_loss": train_loss_history[-1] if train_loss_history else float("inf"),
            "final_val_loss": val_loss_history[-1] if has_validation and val_loss_history else None,
            "effective_epochs": effective_epochs,
        }

        return self, metrics
