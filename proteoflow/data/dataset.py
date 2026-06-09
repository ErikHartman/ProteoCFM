import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from .sampling import adjacent_time_pairs


DEFAULT_METADATA_COLS = {"original_label", "class", "class_name"}


class PCAReducer:

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.fitted_scaler = False
        self.fitted_pca = False
        self.use_pca = None

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        n_features = data.shape[1]

        if self.use_pca is None:
            self.use_pca = self.n_components < n_features

        if self.fitted_scaler is False:
            self.scaler.fit(data)
            self.fitted_scaler = True

        data_scaled = self.scaler.transform(data)

        if self.use_pca:
            print(f"Applying PCA: reducing from {n_features} to {self.n_components} dimensions.")
            if self.fitted_pca is False:
                self.pca.fit(data_scaled)
                self.fitted_pca = True
            data_reduced = self.pca.transform(data_scaled)
            print(f"Total explained variance: {np.sum(self.pca.explained_variance_ratio_)}")
        else:
            print(f"Skipping PCA: n_components ({self.n_components}) >= n_features ({n_features}). Only scaling applied.")
            data_reduced = data_scaled

        return data_reduced

    def inverse_transform(self, data_reduced: np.ndarray) -> np.ndarray:
        if self.fitted_scaler is False:
            raise ValueError("Scaler is not fitted yet.")
        if self.use_pca:
            if self.fitted_pca is False:
                raise ValueError("PCA model is not fitted yet.")
            data_scaled = self.pca.inverse_transform(data_reduced)
        else:
            data_scaled = data_reduced
        return self.scaler.inverse_transform(data_scaled)

    def inverse_transform_velocity(self, velocity_reduced: np.ndarray) -> np.ndarray:
        """Map reduced-space velocity vectors back without adding data means."""
        if self.fitted_scaler is False:
            raise ValueError("Scaler is not fitted yet.")
        if self.use_pca:
            if self.fitted_pca is False:
                raise ValueError("PCA model is not fitted yet.")
            velocity_scaled = velocity_reduced @ self.pca.components_
        else:
            velocity_scaled = velocity_reduced
        return velocity_scaled * self.scaler.scale_


class FlowMatchingDataset(Dataset):
    
    def __init__(
        self,
        df: pd.DataFrame,
        pairs: list[tuple[int, int]],
        feature_cols: list[str],
        covariate_cols: list[str] = None,
        continuous_covariate_cols: list[str] = None,
        batch_ids: np.ndarray = None,
        covariate_categories: dict = None,
        continuous_covariate_stats: dict = None,
    ):
        self.X = df[feature_cols].values
        self.t = df["time"].values
        self.pairs = pairs
        self.batch_ids = batch_ids  # per-pair batch index, or None
        self.covariate_categories = covariate_categories or {}
        self.covariate_category_to_index = {
            name: {value: idx for idx, value in enumerate(categories)}
            for name, categories in self.covariate_categories.items()
        }
        self.continuous_covariate_stats = continuous_covariate_stats or {}
        
        self.covariates = {}
        if covariate_cols:
            for col in covariate_cols:
                self.covariates[col] = df[col].values
        self.continuous_covariates = {}
        if continuous_covariate_cols:
            for col in continuous_covariate_cols:
                self.continuous_covariates[col] = df[col].astype(float).values
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        
        t_interp = self.t[i] + np.random.random() * (self.t[j] - self.t[i])
        covariates = {}
        for name, values in self.covariates.items():
            value = values[i]
            try:
                covariates[name] = self.covariate_category_to_index[name][value]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown or missing covariate value {value!r} for '{name}'."
                ) from exc
        for name, values in self.continuous_covariates.items():
            stats = self.continuous_covariate_stats.get(name, {"mean": 0.0, "std": 1.0})
            value = 0.5 * (float(values[i]) + float(values[j]))
            covariates[name] = (value - stats["mean"]) / stats["std"]
        
        return {
            "x0": self.X[i],
            "x1": self.X[j],
            "t0": float(self.t[i]),
            "t1": float(self.t[j]),
            "t_interp": float(t_interp),
            "covariates": covariates,
        }


def flow_matching_collate_fn(batch):
    result = {
        "x0": torch.tensor(np.stack([b["x0"] for b in batch]), dtype=torch.float32),
        "x1": torch.tensor(np.stack([b["x1"] for b in batch]), dtype=torch.float32),
        "t0": torch.tensor([b["t0"] for b in batch], dtype=torch.float32),
        "t1": torch.tensor([b["t1"] for b in batch], dtype=torch.float32),
        "t_interp": torch.tensor([b["t_interp"] for b in batch], dtype=torch.float32),
    }
    
    if batch[0]["covariates"]:
        result["covariates"] = {}
        for key in batch[0]["covariates"].keys():
            values = [b["covariates"][key] for b in batch]
            dtype = torch.long if isinstance(values[0], (int, np.integer)) else torch.float32
            result["covariates"][key] = torch.tensor(values, dtype=dtype)
    else:
        result["covariates"] = None
    
    return result


class FlowMatchingDataModule:
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] = None,
        covariate_cols: list[str] = None,
        continuous_covariate_cols: list[str] = None,
        n_pca_components: int = None,
        split_col: str = "split",
        n_pairs: int = 1000,
        batch_size: int = 128,
        time_tolerance: float = 1e-6,
        step_sizes: tuple[int, ...] = (1,),
        step_probabilities = None,
        balance_groups: bool = True,
        balance_transitions: bool = True,
        spatial_weight: float = 0.0,
        continuous_covariate_weight: float = 0.0,
        continuous_covariate_bandwidth: float = 1.0,
        metadata_cols: list[str] = None,
    ):
        self.covariate_cols = covariate_cols or []
        self.continuous_covariate_cols = continuous_covariate_cols or []
        all_covariate_cols = self.covariate_cols + self.continuous_covariate_cols
        missing_covariates = [col for col in all_covariate_cols if col not in df.columns]
        if missing_covariates:
            raise ValueError(f"Missing covariate columns: {missing_covariates}")

        null_covariates = [
            col for col in all_covariate_cols
            if df[col].isna().any()
        ]
        if null_covariates:
            raise ValueError(
                "Covariate columns cannot contain missing values: "
                f"{null_covariates}"
            )

        self.covariate_categories = {
            col: sorted(df[col].dropna().unique().tolist())
            for col in self.covariate_cols
        }
        df_train_for_stats = df[df[split_col] == "train"] if split_col in df.columns else df
        self.continuous_covariate_stats = {}
        for col in self.continuous_covariate_cols:
            values = df_train_for_stats[col].astype(float)
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            self.continuous_covariate_stats[col] = {
                "mean": float(values.mean()),
                "std": std,
            }
        self.protein_annotations = df.attrs.get("protein_annotations")
        self.split_col = split_col
        self.n_pairs = n_pairs
        self.batch_size = batch_size
        self.time_tolerance = time_tolerance
        self.step_sizes = step_sizes
        self.step_probabilities = step_probabilities
        self.balance_groups = balance_groups
        self.balance_transitions = balance_transitions
        self.spatial_weight = spatial_weight
        self.continuous_covariate_weight = continuous_covariate_weight
        self.continuous_covariate_bandwidth = continuous_covariate_bandwidth
        self.train_dataset = None
        self.val_dataset = None

        if n_pca_components is not None:
            if feature_cols is not None:
                self.protein_cols = feature_cols
            else:
                meta_cols = (
                    {split_col, "time"}
                    | set(self.covariate_cols)
                    | set(self.continuous_covariate_cols)
                    | DEFAULT_METADATA_COLS
                    | set(metadata_cols or [])
                )
                self.protein_cols = [
                    c for c in df.columns
                    if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
                ]
            if not self.protein_cols:
                raise ValueError("No numeric feature columns available for PCA.")
            missing_features = [col for col in self.protein_cols if col not in df.columns]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")

            pca_reducer = PCAReducer(n_components=n_pca_components)
            # Fit PCA only on train rows to avoid data leakage into validation set
            df_train_only = df[df[split_col] == "train"]
            if df_train_only.empty:
                raise ValueError("Cannot fit PCA because the training split is empty.")
            pca_reducer.transform(df_train_only[self.protein_cols])  # fits scaler + PCA
            X_pca = pca_reducer.transform(df[self.protein_cols])     # applies to all
            pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
            df = df.copy()
            for i, col in enumerate(pca_cols):
                df[col] = X_pca[:, i]
            self.df = df
            self.feature_cols = pca_cols
            self.pca_reducer = pca_reducer
            self.explained_variance_ratio = (
                pca_reducer.pca.explained_variance_ratio_ if pca_reducer.use_pca else None
            )
        else:
            if feature_cols is None:
                raise ValueError("Provide either feature_cols or n_pca_components.")
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")
            self.df = df
            self.feature_cols = feature_cols
            self.pca_reducer = None
            self.protein_cols = None
            self.explained_variance_ratio = None
        
    def setup(self):
        df_train = self.df[self.df[self.split_col] == "train"].reset_index(drop=True)
        df_val = self.df[self.df[self.split_col] == "val"].reset_index(drop=True)
        self.val_dataset = None
        
        train_pairs, train_batch_ids = self._sample_pairs(df_train)
        self.train_dataset = FlowMatchingDataset(
            df_train, train_pairs, self.feature_cols, self.covariate_cols,
            self.continuous_covariate_cols, train_batch_ids,
            covariate_categories=self.covariate_categories,
            continuous_covariate_stats=self.continuous_covariate_stats,
        )
        
        if len(df_val) > 0:
            val_pairs, val_batch_ids = self._sample_pairs(df_val)
            self.val_dataset = FlowMatchingDataset(
                df_val, val_pairs, self.feature_cols, self.covariate_cols,
                self.continuous_covariate_cols, val_batch_ids,
                covariate_categories=self.covariate_categories,
                continuous_covariate_stats=self.continuous_covariate_stats,
            )
    
    def _sample_pairs(self, df):
        pairs, batch_ids = adjacent_time_pairs(
            df,
            self.feature_cols,
            self.covariate_cols,
            self.n_pairs,
            time_tolerance=self.time_tolerance,
            step_sizes=self.step_sizes,
            step_probabilities=self.step_probabilities,
            balance_groups=self.balance_groups,
            balance_transitions=self.balance_transitions,
            spatial_weight=self.spatial_weight,
            continuous_covariate_cols=self.continuous_covariate_cols,
            continuous_covariate_stats=self.continuous_covariate_stats,
            continuous_covariate_weight=self.continuous_covariate_weight,
            continuous_covariate_bandwidth=self.continuous_covariate_bandwidth,
        )
        return pairs, batch_ids
    
    def resample_train_pairs(self):
        """Re-draw training pairs (useful to call once per epoch for more diversity)."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before resampling pairs")
        df_train = self.df[self.df[self.split_col] == "train"].reset_index(drop=True)
        new_pairs, new_batch_ids = self._sample_pairs(df_train)
        self.train_dataset.pairs = new_pairs
        self.train_dataset.batch_ids = new_batch_ids

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=flow_matching_collate_fn,
            num_workers=0
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        if len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=flow_matching_collate_fn,
            num_workers=0
        )
    
    @property
    def n_features(self) -> int:
        """Number of feature dimensions in the data."""
        return len(self.feature_cols)
    
    @property
    def n_covariates(self) -> int:
        """Number of covariate columns."""
        return len(self.covariate_cols) + len(self.continuous_covariate_cols)
