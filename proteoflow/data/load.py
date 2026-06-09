import pandas as pd
import numpy as np
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ACCESSION_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
COVID_WHO_GRADE_MIN = 3.0
COVID_WHO_GRADE_MAX = 7.0


def _with_train_val_split(df, test_size, random_state, stratify=None):
    df = df.copy()
    df["split"] = "train"
    if test_size is None or test_size == 0:
        return df

    _, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"
    return df


def _covid_grade_to_time(who_grade):
    who_grade = pd.Series(who_grade).astype(float).replace(0.0, COVID_WHO_GRADE_MIN)
    invalid = ~who_grade.between(COVID_WHO_GRADE_MIN, COVID_WHO_GRADE_MAX)
    if invalid.any():
        invalid_values = sorted(who_grade.loc[invalid].dropna().unique().tolist())
        raise ValueError(
            "COVID WHO grades must be 0 or in the 3-7 interval before normalization; "
            f"found {invalid_values[:5]}."
        )
    return (who_grade - COVID_WHO_GRADE_MIN) / (COVID_WHO_GRADE_MAX - COVID_WHO_GRADE_MIN)


def _chunked(values, size):
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def _looks_like_uniprot_accession(identifier: str) -> bool:
    return bool(UNIPROT_ACCESSION_PATTERN.fullmatch(str(identifier)))


def _extract_protein_name(description: dict) -> str | None:
    if not description:
        return None

    queue = [description]
    while queue:
        item = queue.pop(0)
        if isinstance(item, dict):
            full_name = item.get("fullName")
            if isinstance(full_name, dict) and full_name.get("value"):
                return full_name["value"]
            if isinstance(full_name, str):
                return full_name
            for key in ("recommendedName", "submissionNames", "alternativeNames", "contains", "includes"):
                value = item.get(key)
                if isinstance(value, list):
                    queue.extend(value)
                elif isinstance(value, dict):
                    queue.append(value)
        elif isinstance(item, list):
            queue.extend(item)
    return None


def _fetch_uniprot_annotations(accessions: list[str], timeout: float = 30.0) -> list[dict]:
    records = []
    for accession_chunk in _chunked(accessions, 100):
        query = " OR ".join(f"accession:{accession}" for accession in accession_chunk)
        params = urlencode(
            {
                "query": query,
                "format": "json",
                "size": len(accession_chunk),
            }
        )
        request = Request(
            f"{UNIPROT_SEARCH_URL}?{params}",
            headers={"User-Agent": "proteoflow/0.1.0"},
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        for entry in payload.get("results", []):
            genes = entry.get("genes") or []
            gene_name = None
            for gene in genes:
                gene_name = gene.get("geneName", {}).get("value")
                if gene_name:
                    break

            records.append(
                {
                    "source_id": entry.get("primaryAccession"),
                    "uniprot_accession": entry.get("primaryAccession"),
                    "gene_name": gene_name,
                    "protein_name": _extract_protein_name(entry.get("proteinDescription") or {}),
                }
            )
    return records


def _build_annotation_rows(
    protein_ids: list[str],
    fetched_df: pd.DataFrame | None,
    label_preference: str = "gene_name",
) -> pd.DataFrame:
    fetched_df = fetched_df.copy() if fetched_df is not None else pd.DataFrame()
    if fetched_df.empty:
        fetched_df = pd.DataFrame(columns=["source_id", "uniprot_accession", "gene_name", "protein_name"])

    fetched_df = fetched_df.drop_duplicates(subset=["source_id"], keep="first")
    fetched_lookup = fetched_df.set_index("source_id").to_dict(orient="index")

    rows = []
    base_labels = []
    for protein_id in protein_ids:
        record = fetched_lookup.get(protein_id, {})
        gene_name = record.get("gene_name")
        protein_name = record.get("protein_name")
        uniprot_accession = record.get("uniprot_accession", protein_id if _looks_like_uniprot_accession(protein_id) else None)

        if _looks_like_uniprot_accession(protein_id):
            base_label = gene_name or protein_name or protein_id
        else:
            base_label = protein_id
            gene_name = gene_name or protein_id

        base_labels.append(str(base_label).strip())
        rows.append(
            {
                "source_id": protein_id,
                "uniprot_accession": uniprot_accession,
                "gene_name": gene_name,
                "protein_name": protein_name,
            }
        )

    label_counts = Counter(base_labels)
    for row, base_label in zip(rows, base_labels):
        display_name = base_label if label_counts[base_label] == 1 else f"{base_label}|{row['source_id']}"
        row["display_name"] = display_name
        preferred_label = row.get(label_preference) or row.get("protein_name") or row["source_id"]
        row["preferred_label"] = preferred_label

    return pd.DataFrame(rows)


def annotate_protein_columns(
    df: pd.DataFrame,
    *,
    cache_path: str | Path,
    annotate: bool = True,
    fetch_missing: bool = True,
    refresh_cache: bool = False,
    label_preference: str = "gene_name",
) -> pd.DataFrame:
    df = df.copy()
    protein_ids = list(df.columns)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached_df = pd.DataFrame(columns=["source_id", "uniprot_accession", "gene_name", "protein_name", "display_name", "preferred_label"])
    if cache_path.exists() and not refresh_cache:
        cached_df = pd.read_csv(cache_path).fillna("")

    accession_ids = [protein_id for protein_id in protein_ids if _looks_like_uniprot_accession(protein_id)]
    missing_ids = sorted(set(accession_ids).difference(set(cached_df.get("source_id", []))))

    if annotate and fetch_missing and missing_ids:
        try:
            fetched_records = _fetch_uniprot_annotations(missing_ids)
            fetched_df = pd.DataFrame(fetched_records)
            if not fetched_df.empty:
                cached_df = (
                    pd.concat([cached_df, fetched_df], ignore_index=True)
                    .drop_duplicates(subset=["source_id"], keep="last")
                )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            warnings.warn(
                f"Could not fetch UniProt annotations ({exc}). Falling back to raw protein identifiers.",
                stacklevel=2,
            )

    annotations_df = _build_annotation_rows(protein_ids, cached_df, label_preference=label_preference)
    annotations_df.to_csv(cache_path, index=False)

    rename_map = dict(zip(annotations_df["source_id"], annotations_df["display_name"]))
    df = df.rename(columns=rename_map)
    df.attrs["protein_annotations"] = annotations_df
    df.attrs["protein_rename_map"] = rename_map
    return df


def load_sepsis(
    data_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    include_plasma: bool = True,
    covariates: list[str] = None,
    annotate_proteins: bool = True,
    fetch_protein_annotations: bool = True,
    refresh_protein_annotations: bool = False,
):
    covariates = ["tissue", "model"] if covariates is None else covariates
    suffix = "" if include_plasma else "_no_plasma"
    data = pd.read_csv(
        f"{data_dir}/mouse_sepsis_proteomics_data_batch_corrected{suffix}.csv"
    )
    metadata = pd.read_csv(
        f"{data_dir}/mouse_sepsis_proteomics_metadata_batch_corrected{suffix}.csv"
    )

    data.set_index("Run", inplace=True)
    metadata.set_index("Run", inplace=True)
    metadata = metadata.reindex(data.index)
    if metadata.isnull().any().any():
        missing_runs = metadata.index[metadata.isnull().any(axis=1)].tolist()
        raise ValueError(f"Missing sepsis metadata for runs: {missing_runs[:5]}")

    nhpi = metadata["nhpi"].to_numpy()

    tissue_cat = metadata["Tissue"].astype("category")
    model_cat = metadata["Model"].astype("category")

    df = data.copy()
    df = annotate_protein_columns(
        df,
        cache_path=Path(data_dir) / "sepsis_uniprot_annotations.csv",
        annotate=annotate_proteins,
        fetch_missing=fetch_protein_annotations,
        refresh_cache=refresh_protein_annotations,
    )
    df["time"] = nhpi

    if "tissue" in covariates:
        df["tissue"] = tissue_cat.values

    if "model" in covariates:
        df["model"] = model_cat.values

    stratify = (
        pd.qcut(nhpi, q=5, duplicates="drop")
        if test_size is not None and test_size > 0
        else None
    )
    return _with_train_val_split(df, test_size, random_state, stratify)


def load_covid(
    data_dir: str = "data/covid",
    test_size: float = 0.1,
    random_state: int = 42,
    covariates: list[str] = None,
    annotate_proteins: bool = True,
    fetch_protein_annotations: bool = True,
    refresh_protein_annotations: bool = False,
):
    quant = pd.read_csv(f"{data_dir}/quant.csv")
    meta = pd.read_csv(f"{data_dir}/meta.csv")

    quant.set_index("File", inplace=True)
    meta.set_index("File", inplace=True)
    meta = meta.reindex(quant.index)
    required_metadata_cols = ["WHO grade"]
    missing_required = meta[required_metadata_cols].isnull().any(axis=1)
    if missing_required.any():
        missing_files = meta.index[missing_required].tolist()
        raise ValueError(f"Missing required COVID metadata for files: {missing_files[:5]}")

    who_grade = meta["WHO grade"].astype(float).replace(0.0, COVID_WHO_GRADE_MIN)
    who_grade_norm = _covid_grade_to_time(who_grade)
    meta["WHO grade"] = who_grade.values

    df = quant.copy()
    df = annotate_protein_columns(
        df,
        cache_path=Path(data_dir) / "covid_uniprot_annotations.csv",
        annotate=annotate_proteins,
        fetch_missing=fetch_protein_annotations,
        refresh_cache=refresh_protein_annotations,
    )
    df["time"] = who_grade_norm.values

    raw_meta = None
    for covariate in covariates or []:
        if covariate in meta.columns:
            values = meta[covariate]
        elif covariate == "age":
            if raw_meta is None:
                raw_meta_path = Path(data_dir) / "raw" / "prot_quant_clin_meas_meta.tsv"
                if not raw_meta_path.exists():
                    raise ValueError("COVID age covariate requires raw/prot_quant_clin_meas_meta.tsv.")
                raw_meta = pd.read_csv(raw_meta_path, sep="\t").set_index("File")
                raw_meta = raw_meta.reindex(quant.index)
            values = raw_meta["Age"]
        else:
            raise ValueError(f"COVID metadata does not contain covariate {covariate!r}.")
        if covariate == "age_group":
            values = values.astype("category")
        df[covariate] = values.values

    if covariates:
        missing_covariate_mask = df[covariates].isna().any(axis=1)
        if missing_covariate_mask.any():
            n_missing = int(missing_covariate_mask.sum())
            warnings.warn(
                f"Dropping {n_missing} COVID samples with missing requested covariates: {covariates}.",
                stacklevel=2,
            )
            df = df.loc[~missing_covariate_mask].copy()

    stratify = (
        pd.qcut(df["time"], q=5, duplicates="drop")
        if test_size is not None and test_size > 0
        else None
    )
    return _with_train_val_split(df, test_size, random_state, stratify)


def load_synthetic(
    num_points: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
    noise_std: float = 0.1,
    mode: str = "discrete",
    type: str = "circle",
):
    np.random.seed(random_state)

    if type == "gaussians":
        start_center = np.array([-1.5, 0.0], dtype=float)
        upper_end_center = np.array([1.5, 1.25], dtype=float)
        lower_end_center = np.array([1.5, -1.25], dtype=float)

        if mode == "discrete":
            upper_start = start_center + np.random.normal(0, noise_std, size=(num_points, 2))
            upper_end = upper_end_center + np.random.normal(0, noise_std, size=(num_points, 2))
            lower_start = start_center + np.random.normal(0, noise_std, size=(num_points, 2))
            lower_end = lower_end_center + np.random.normal(0, noise_std, size=(num_points, 2))

            points = np.vstack([upper_start, upper_end, lower_start, lower_end])
            labels = np.array(
                [0] * num_points + [1] * num_points + [2] * num_points + [3] * num_points
            )
            classes = np.array([0] * (2 * num_points) + [1] * (2 * num_points))
            time = np.array(
                [0.0] * num_points
                + [1.0] * num_points
                + [0.0] * num_points
                + [1.0] * num_points
            )
        elif mode == "continuous":
            upper_time = np.random.uniform(0, 1, 2 * num_points)
            lower_time = np.random.uniform(0, 1, 2 * num_points)

            upper_means = (
                (1 - upper_time)[:, None] * start_center
                + upper_time[:, None] * upper_end_center
            )
            lower_means = (
                (1 - lower_time)[:, None] * start_center
                + lower_time[:, None] * lower_end_center
            )

            upper_points = upper_means + np.random.normal(0, noise_std, size=(2 * num_points, 2))
            lower_points = lower_means + np.random.normal(0, noise_std, size=(2 * num_points, 2))

            points = np.vstack([upper_points, lower_points])
            labels = np.array([0] * (2 * num_points) + [1] * (2 * num_points))
            classes = np.array([0] * (2 * num_points) + [1] * (2 * num_points))
            time = np.concatenate([upper_time, lower_time])
        else:
            raise ValueError(f"Unknown synthetic mode: {mode!r}")

        df = pd.DataFrame(points, columns=["PC1", "PC2"])
        df["time"] = time
        df["original_label"] = labels
        df["class"] = classes
        df["class_name"] = ["upper" if c == 0 else "lower" for c in classes]
        return _with_train_val_split(df, test_size, random_state)

    if type == "tree":
        t0_samples = np.random.uniform(0, 0.5, num_points)
        t1_samples = np.random.uniform(0.5, 1.0, num_points)
    elif mode == "continuous":
        all_t = np.random.uniform(0, 1, 2 * num_points)
        t0_samples = all_t[:num_points]
        t1_samples = all_t[num_points:]
    else:
        t0_samples = np.abs(np.random.normal(0, 1 / (2 * np.pi), num_points))
        t1_samples = 1 - np.abs(np.random.normal(0, 1 / (2 * np.pi), num_points))

    angles0 = np.pi * (1 - t0_samples)
    angles1 = np.pi * (1 - t1_samples)

    noise0 = np.random.normal(0, noise_std, num_points)
    noise1 = np.random.normal(0, noise_std, num_points)

    x0 = (1 + noise0) * np.cos(angles0)
    y0 = (1.25 + noise0) * np.sin(angles0)
    x1 = (1 + noise1) * np.cos(angles1)
    y1 = (1.25 + noise1) * np.sin(angles1)

    if type == "arch":
        points = np.vstack([np.column_stack((x0, y0)), np.column_stack((x1, y1))])
        labels = np.array([0] * num_points + [1] * num_points)

        if mode == "discrete":
            time = np.array([0.0] * num_points + [1.0] * num_points)
        else:
            time = np.concatenate([t0_samples, t1_samples])

        df = pd.DataFrame(points, columns=["PC1", "PC2"])
        df["time"] = time
        df["original_label"] = labels

    elif type == "circle":
        y0_neg = -(1.25 + np.random.normal(0, noise_std, num_points)) * np.sin(angles0)
        y1_neg = -(1.25 + np.random.normal(0, noise_std, num_points)) * np.sin(angles1)

        points = np.vstack(
            [
                np.column_stack((x0, y0)),
                np.column_stack((x1, y1)),
                np.column_stack((x0, y0_neg)),
                np.column_stack((x1, y1_neg)),
            ]
        )
        labels = np.array(
            [0] * num_points + [1] * num_points + [2] * num_points + [3] * num_points
        )
        classes = np.array([0] * (2 * num_points) + [1] * (2 * num_points))

        if mode == "discrete":
            time = np.array(
                [0.0] * num_points
                + [1.0] * num_points
                + [0.0] * num_points
                + [1.0] * num_points
            )
        else:
            time = np.concatenate([t0_samples, t1_samples, t0_samples, t1_samples])

        df = pd.DataFrame(points, columns=["PC1", "PC2"])
        df["time"] = time
        df["original_label"] = labels
        df["class"] = classes
        df["class_name"] = ["upper" if c == 0 else "lower" for c in classes]

    elif type == "tree":
        y1_neg = -(1.25 + np.random.normal(0, noise_std, num_points)) * np.sin(angles1)
        y_shift = 2.5

        points = np.vstack(
            [
                np.column_stack((x0, y0)),
                np.column_stack((x1, y1)),
                np.column_stack((x0, y0)),
                np.column_stack((x1, y1_neg + y_shift)),
            ]
        )
        labels = np.array(
            [0] * num_points + [1] * num_points + [2] * num_points + [3] * num_points
        )
        classes = np.array([0] * (2 * num_points) + [1] * (2 * num_points))

        if mode == "discrete":
            time = np.array(
                [0.0] * num_points
                + [1.0] * num_points
                + [0.0] * num_points
                + [1.0] * num_points
            )
        else:
            time = np.concatenate([t0_samples, t1_samples, t0_samples, t1_samples])

        df = pd.DataFrame(points, columns=["PC1", "PC2"])
        df["time"] = time
        df["original_label"] = labels
        df["class"] = classes
        df["class_name"] = ["upper" if c == 0 else "lower" for c in classes]

    else:
        raise ValueError(f"Unknown synthetic type: {type!r}")

    return _with_train_val_split(df, test_size, random_state)
