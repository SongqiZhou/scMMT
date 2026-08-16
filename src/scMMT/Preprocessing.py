"""Preprocessing pipeline described in the scMMT paper."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from anndata import concat as ad_concat
from sklearn.decomposition import FactorAnalysis, TruncatedSVD

from .Utils import make_dense

try:  # Intel acceleration is useful but not required on every platform.
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:  # pragma: no cover - depends on the installation platform
    pass


sc.settings.verbosity = 0


def _nonzero_counts(matrix, axis: int) -> np.ndarray:
    return np.asarray((matrix > 1e-8).sum(axis=axis)).reshape(-1)


def _validate_inputs(gene_trainsets, protein_trainsets, train_batchkeys):
    if not isinstance(gene_trainsets, list) or not gene_trainsets:
        raise TypeError("gene_trainsets must be a non-empty list of AnnData objects")
    if protein_trainsets is not None and not isinstance(protein_trainsets, list):
        raise TypeError("protein_trainsets must be a list or None")
    if protein_trainsets and len(protein_trainsets) != len(gene_trainsets):
        raise ValueError("Provide one protein AnnData object per RNA reference dataset")
    if train_batchkeys is not None and len(train_batchkeys) != len(gene_trainsets):
        raise ValueError("train_batchkeys must match the number of reference datasets")


def preprocess(
    gene_trainsets,
    protein_trainsets=None,
    gene_test=None,
    train_batchkeys=None,
    test_batchkey=None,
    type_key=None,
    gene_list=None,
    select_hvg=True,
    cell_normalize=True,
    log_normalize=True,
    gene_normalize=True,
    min_cells=1,
    min_genes=1,
    n_svd=300,
    n_fa=180,
    n_hvg=550,
    dataset_batch=True,
    data_dir="data.pkl",
    data_load=False,
    seed=5,
):
    """Prepare reference/query data and return the tensors consumed by scMMT.

    Cached pickle files must only be loaded from trusted sources.  Passing
    ``data_dir=None`` disables both cache loading and cache creation.
    """

    _validate_inputs(gene_trainsets, protein_trainsets, train_batchkeys)
    if min_cells < 0 or min_genes < 0:
        raise ValueError("min_cells and min_genes cannot be negative")
    if n_svd <= 0 or n_fa <= 0 or n_hvg <= 0:
        raise ValueError("n_svd, n_fa, and n_hvg must be positive")

    if data_load:
        if data_dir is None:
            raise ValueError("data_dir is required when data_load=True")
        with Path(data_dir).open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - explicitly documented trusted cache

    # Avoid mutating caller-owned AnnData objects and mutable default lists.
    gene_trainsets = [adata.copy() for adata in gene_trainsets]
    gene_test = None if gene_test is None else gene_test.copy()
    requested_genes = set(gene_list or [])

    if protein_trainsets:
        protein_trainsets = [adata.copy() for adata in protein_trainsets]
    else:
        protein_trainsets = [
            AnnData(
                np.zeros((adata.n_obs, 0), dtype=np.float32),
                obs=adata.obs.copy(),
            )
            for adata in gene_trainsets
        ]

    for index, (genes, proteins) in enumerate(zip(gene_trainsets, protein_trainsets)):
        if not genes.obs_names.equals(proteins.obs_names):
            raise ValueError(f"RNA and protein cell indices differ for reference dataset {index}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    if type_key is not None and not all(
        type_key in adata.obs.columns for adata in protein_trainsets
    ):
        raise KeyError(f"Cell-type column {type_key!r} is missing from protein metadata")

    for index, (genes, proteins) in enumerate(zip(gene_trainsets, protein_trainsets), 1):
        if train_batchkeys is None:
            batch = pd.Series(f"DS-{index}", index=genes.obs_names)
        else:
            key = train_batchkeys[index - 1]
            if key not in genes.obs or key not in proteins.obs:
                raise KeyError(f"Batch column {key!r} is missing from reference dataset {index}")
            batch = "DS-" + str(index) + " " + genes.obs[key].astype(str)
        genes.obs["batch"] = batch.to_numpy()
        proteins.obs["batch"] = batch.to_numpy()
        genes.obs["Dataset"] = f"Dataset {index}"
        proteins.obs["Dataset"] = f"Dataset {index}"

    if gene_test is not None:
        if test_batchkey is None:
            gene_test.obs["batch"] = "DS-Test"
        else:
            if test_batchkey not in gene_test.obs:
                raise KeyError(f"Batch column {test_batchkey!r} is missing from gene_test")
            gene_test.obs["batch"] = (
                "DS-Test " + gene_test.obs[test_batchkey].astype(str)
            ).to_numpy()

    if min_genes:
        print("\nQC Filtering Training Cells")
        for index in range(len(gene_trainsets)):
            keep = _nonzero_counts(gene_trainsets[index].X, axis=1) >= min_genes
            gene_trainsets[index] = gene_trainsets[index][keep].copy()
            protein_trainsets[index] = protein_trainsets[index][keep].copy()
        if gene_test is not None:
            print("QC Filtering Testing Cells")
            gene_test = gene_test[_nonzero_counts(gene_test.X, axis=1) >= min_genes].copy()

    if any(adata.n_obs == 0 for adata in gene_trainsets):
        raise ValueError("QC filtering removed every cell from a reference dataset")
    if gene_test is not None and gene_test.n_obs == 0:
        raise ValueError("QC filtering removed every query cell")

    if type_key is not None:
        categories = {}
        for dataset in protein_trainsets:
            if dataset.obs[type_key].isna().any():
                raise ValueError(f"Cell-type column {type_key!r} contains missing values")
            for celltype in dataset.obs[type_key]:
                if celltype not in categories:
                    categories[celltype] = len(categories)
    else:
        categories = None

    if min_cells:
        print("\nQC Filtering Training Genes")
        for index, adata in enumerate(gene_trainsets):
            expressed = set(adata.var_names[_nonzero_counts(adata.X, axis=0) >= min_cells])
            selected = sorted(expressed | (requested_genes & set(adata.var_names)))
            gene_trainsets[index] = adata[:, selected].copy()
        if gene_test is not None:
            print("QC Filtering Testing Genes")
            expressed = set(gene_test.var_names[_nonzero_counts(gene_test.X, axis=0) >= min_cells])
            selected = sorted(expressed | (requested_genes & set(gene_test.var_names)))
            gene_test = gene_test[:, selected].copy()

    for genes, proteins in zip(gene_trainsets, protein_trainsets):
        genes.layers["raw"] = genes.X.copy()
        proteins.layers["raw"] = proteins.X.copy()
    if gene_test is not None:
        gene_test.layers["raw"] = gene_test.X.copy()

    if cell_normalize:
        print("\nNormalizing Training Cells")
        for adata in gene_trainsets:
            sc.pp.normalize_total(adata)
        if gene_test is not None:
            print("Normalizing Testing Cells")
            sc.pp.normalize_total(gene_test, key_added="scale_factor")

    if log_normalize:
        print("\nLog-Normalizing Training Data")
        for adata in gene_trainsets:
            sc.pp.log1p(adata)
        if gene_test is not None:
            print("Log-Normalizing Testing Data")
            sc.pp.log1p(gene_test)

    gene_train = ad_concat(
        gene_trainsets,
        axis=0,
        join="inner",
        merge="same",
        index_unique=None,
    )
    if gene_train.n_vars == 0:
        raise ValueError("Reference datasets do not share any genes")

    if gene_test is not None:
        common_genes = gene_train.var_names.intersection(gene_test.var_names)
        if len(common_genes) == 0:
            raise ValueError("Reference and query datasets do not share any genes")
        gene_train = gene_train[:, common_genes].copy()
        gene_test = gene_test[:, common_genes].copy()

    make_dense(gene_train)
    for adata in protein_trainsets:
        make_dense(adata)
    if gene_test is not None:
        make_dense(gene_test)

    if gene_test is None:
        feature_matrix = np.asarray(gene_train.X)
        train_rows = gene_train.n_obs
    else:
        feature_matrix = np.concatenate((np.asarray(gene_train.X), np.asarray(gene_test.X)), axis=0)
        train_rows = gene_train.n_obs
        if dataset_batch:
            combined = AnnData(feature_matrix)
            gene_train.obs["dataset"] = "train"
            gene_test.obs["dataset"] = "test"
            combined.obs = pd.concat([gene_train.obs, gene_test.obs], axis=0)
            print("\nComBat batch correction")
            sc.pp.combat(combined, key="dataset")
            feature_matrix = np.asarray(combined.X)

    max_svd = max(1, min(feature_matrix.shape[1] - 1, feature_matrix.shape[0] - 1))
    max_fa = max(1, min(feature_matrix.shape[0], feature_matrix.shape[1]))
    effective_svd = min(n_svd, max_svd)
    effective_fa = min(n_fa, max_fa)

    print("\nTSVD...")
    svd_features = TruncatedSVD(n_components=effective_svd, random_state=seed).fit_transform(
        feature_matrix
    )
    print("\nFactor analysis...")
    fa_features = FactorAnalysis(n_components=effective_fa, random_state=seed).fit_transform(
        feature_matrix
    )

    gene_train.obsm["X_svd"] = svd_features[:train_rows]
    gene_train.obsm["X_fa"] = fa_features[:train_rows]
    if gene_test is not None:
        gene_test.obsm["X_svd"] = svd_features[train_rows:]
        gene_test.obsm["X_fa"] = fa_features[train_rows:]

    if select_hvg:
        print("\nFinding HVGs")
        tmp = (
            ad_concat([gene_train, gene_test], join="inner", index_unique=None)
            if gene_test is not None
            else gene_train.copy()
        )
        if not cell_normalize or not log_normalize:
            print("Warning: HVG selection is most reliable after cell and log normalization")
        if len(tmp) > 100_000:
            tmp = tmp[rng.choice(len(tmp), 100_000, replace=False)].copy()
        sc.pp.highly_variable_genes(
            tmp,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_bins=20,
            subset=False,
            batch_key="batch",
            n_top_genes=min(n_hvg, tmp.n_vars),
        )
        selected_genes = set(tmp.var_names[tmp.var["highly_variable"]])
        selected_genes.update(requested_genes & set(gene_train.var_names))
        selected_genes = sorted(selected_genes)
        gene_train = gene_train[:, selected_genes].copy()
        if gene_test is not None:
            gene_test = gene_test[:, selected_genes].copy()

    if gene_normalize:
        print("\nNormalizing Gene Training Data by Batch")
        for batch in gene_train.obs["batch"].unique():
            indices = np.asarray(gene_train.obs["batch"] == batch)
            subset = gene_train[indices].copy()
            sc.pp.scale(subset)
            gene_train.X[indices, :] = subset.X
        if gene_test is not None:
            print("\nNormalizing Gene Testing Data by Batch")
            for batch in gene_test.obs["batch"].unique():
                indices = np.asarray(gene_test.obs["batch"] == batch)
                subset = gene_test[indices].copy()
                sc.pp.scale(subset)
                gene_test.X[indices, :] = subset.X

    # Protein panels are merged regardless of whether gene scaling is enabled.
    protein_sets = [set(adata.var_names) for adata in protein_trainsets]
    protein_train = ad_concat(
        protein_trainsets,
        axis=0,
        join="outer",
        merge="same",
        fill_value=0.0,
        index_unique=None,
    )
    train_keys = np.cumsum([adata.n_obs for adata in protein_trainsets])[:-1].tolist()
    protein_masks = np.asarray(
        [
            [int(protein in measured) for protein in protein_train.var_names]
            for measured in protein_sets
        ],
        dtype=np.float32,
    )
    for index, mask in enumerate(protein_masks, 1):
        protein_train.var[f"Dataset {index}"] = mask.astype(bool)

    gene_train.obsm["result"] = np.concatenate(
        [gene_train.obsm["X_svd"], gene_train.obsm["X_fa"], gene_train.X],
        axis=1,
    ).astype(np.float32, copy=False)
    if gene_test is not None:
        gene_test.obsm["result"] = np.concatenate(
            [gene_test.obsm["X_svd"], gene_test.obsm["X_fa"], gene_test.X],
            axis=1,
        ).astype(np.float32, copy=False)
    protein_train.obsm["result"] = np.asarray(protein_train.X, dtype=np.float32)

    data = (
        gene_train,
        protein_train,
        gene_test,
        protein_masks,
        train_keys,
        categories,
    )
    if data_dir is not None:
        cache_path = Path(data_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(data, handle)
    return data
