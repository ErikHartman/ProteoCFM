import numpy as np
import pandas as pd
from typing import Optional


def _group_indices_by_covariates(
    df: pd.DataFrame,
    covariate_cols: list[str],
):
    if covariate_cols:
        cov_groups = {}
        for idx in range(len(df)):
            cov_tuple = tuple(df.iloc[idx][col] for col in covariate_cols)
            cov_groups.setdefault(cov_tuple, []).append(idx)
    else:
        cov_groups = {"all": list(range(len(df)))}
    return {
        key: np.asarray(indices, dtype=int)
        for key, indices in cov_groups.items()
    }


def _build_time_buckets(
    indices: np.ndarray,
    time_values: np.ndarray,
    tolerance: float,
):
    if len(indices) == 0:
        return []

    sorted_indices = indices[np.argsort(time_values[indices], kind="stable")]
    buckets = []
    current_bucket = [int(sorted_indices[0])]
    current_time = time_values[sorted_indices[0]]

    for idx in sorted_indices[1:]:
        idx = int(idx)
        t_val = time_values[idx]
        if abs(t_val - current_time) <= tolerance:
            current_bucket.append(idx)
        else:
            buckets.append(np.asarray(current_bucket, dtype=int))
            current_bucket = [idx]
            current_time = t_val

    buckets.append(np.asarray(current_bucket, dtype=int))
    return buckets


def adjacent_time_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    covariate_cols: list[str],
    n_pairs: int,
    time_tolerance: float = 1e-6,
    step_sizes: tuple[int, ...] = (1,),
    step_probabilities: Optional[tuple[float, ...]] = None,
    balance_groups: bool = True,
    balance_transitions: bool = True,
    spatial_weight: float = 0.0,
    continuous_covariate_cols: Optional[list[str]] = None,
    continuous_covariate_stats: Optional[dict[str, dict[str, float]]] = None,
    continuous_covariate_weight: float = 0.0,
    continuous_covariate_bandwidth: float = 1.0,
    seed: Optional[int] = None,
):
    if n_pairs <= 0:
        return [], np.array([], dtype=int)

    if spatial_weight < 0 or spatial_weight > 1:
        raise ValueError("spatial_weight must be between 0 and 1.")
    if continuous_covariate_weight < 0 or continuous_covariate_weight > 1:
        raise ValueError("continuous_covariate_weight must be between 0 and 1.")
    if continuous_covariate_bandwidth <= 0:
        raise ValueError("continuous_covariate_bandwidth must be positive.")

    step_sizes = tuple(int(step) for step in step_sizes)
    if not step_sizes or any(step < 1 for step in step_sizes):
        raise ValueError("step_sizes must contain positive integers.")

    if step_probabilities is None:
        step_weights = {
            step: 1.0 for step in step_sizes
        }
    else:
        if len(step_probabilities) != len(step_sizes):
            raise ValueError("step_probabilities must match step_sizes in length.")
        if any(prob < 0 for prob in step_probabilities):
            raise ValueError("step_probabilities cannot contain negative values.")
        total_prob = float(sum(step_probabilities))
        if total_prob <= 0:
            raise ValueError("step_probabilities must sum to a positive value.")
        step_weights = {
            step: prob / total_prob
            for step, prob in zip(step_sizes, step_probabilities)
        }

    rng = np.random.default_rng(seed)
    t = df["time"].to_numpy()
    X = df[feature_cols].to_numpy() if spatial_weight > 0 else None
    continuous_covariate_cols = continuous_covariate_cols or []
    continuous_covariates = None
    if continuous_covariate_cols and continuous_covariate_weight > 0:
        continuous_covariate_stats = continuous_covariate_stats or {}
        continuous_covariates = df[continuous_covariate_cols].astype(float).to_numpy()
        for col_idx, col in enumerate(continuous_covariate_cols):
            stats = continuous_covariate_stats.get(col, {"mean": 0.0, "std": 1.0})
            std = stats["std"] if stats["std"] > 0 else 1.0
            continuous_covariates[:, col_idx] = (
                continuous_covariates[:, col_idx] - stats["mean"]
            ) / std

    transitions = []
    group_indices = _group_indices_by_covariates(df, covariate_cols)
    next_batch_id = 0

    for group_key, indices in group_indices.items():
        buckets = _build_time_buckets(indices, t, tolerance=time_tolerance)
        if len(buckets) < 2:
            continue

        for step in step_sizes:
            if step >= len(buckets):
                continue

            for bucket_idx in range(len(buckets) - step):
                src = buckets[bucket_idx]
                tgt = buckets[bucket_idx + step]
                if len(src) == 0 or len(tgt) == 0:
                    continue

                transitions.append(
                    {
                        "group_key": group_key,
                        "src": src,
                        "tgt": tgt,
                        "step": step,
                        "pair_count": len(src) * len(tgt),
                        "batch_id": next_batch_id,
                    }
                )
                next_batch_id += 1

    if not transitions:
        return [], np.array([], dtype=int)

    base_weights = np.asarray(
        [
            step_weights[transition["step"]]
            * (1.0 if balance_transitions else float(transition["pair_count"]))
            for transition in transitions
        ],
        dtype=float,
    )

    if balance_groups:
        group_totals = {}
        for transition, weight in zip(transitions, base_weights):
            group_key = transition["group_key"]
            group_totals[group_key] = group_totals.get(group_key, 0.0) + weight

        transition_weights = np.asarray(
            [
                weight / group_totals[transition["group_key"]]
                for transition, weight in zip(transitions, base_weights)
            ],
            dtype=float,
        )
    else:
        transition_weights = base_weights

    transition_probabilities = transition_weights / transition_weights.sum()
    selected_transition_indices = rng.choice(
        len(transitions),
        size=n_pairs,
        replace=True,
        p=transition_probabilities,
    )

    pairs = []
    batch_ids = []
    for transition_idx in selected_transition_indices:
        transition = transitions[int(transition_idx)]
        src = transition["src"]
        tgt = transition["tgt"]

        src_index = int(rng.choice(src))
        if len(tgt) > 1:
            target_probabilities = np.full(len(tgt), 1.0 / len(tgt))

            if spatial_weight > 0 and X is not None:
                distances = np.linalg.norm(X[tgt] - X[src_index], axis=1)
                spatial_scores = np.exp(-distances)
                spatial_sum = spatial_scores.sum()
                if np.isfinite(spatial_sum) and spatial_sum > 0:
                    spatial_probabilities = spatial_scores / spatial_sum
                    target_probabilities = (
                        (1.0 - spatial_weight) * target_probabilities
                        + spatial_weight * spatial_probabilities
                    )

            if continuous_covariates is not None:
                covariate_distances = np.linalg.norm(
                    continuous_covariates[tgt] - continuous_covariates[src_index],
                    axis=1,
                )
                covariate_scores = np.exp(
                    -covariate_distances / continuous_covariate_bandwidth
                )
                covariate_sum = covariate_scores.sum()
                if np.isfinite(covariate_sum) and covariate_sum > 0:
                    covariate_probabilities = covariate_scores / covariate_sum
                    target_probabilities = (
                        (1.0 - continuous_covariate_weight) * target_probabilities
                        + continuous_covariate_weight * covariate_probabilities
                    )

            target_probabilities = target_probabilities / target_probabilities.sum()
            target_index = int(rng.choice(tgt, p=target_probabilities))
        else:
            target_index = int(rng.choice(tgt))

        pairs.append((src_index, target_index))
        batch_ids.append(int(transition["batch_id"]))

    return pairs, np.asarray(batch_ids, dtype=int)
