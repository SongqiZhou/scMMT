"""Small public utilities for scMMT."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix, issparse


def build_dir(dir_path):
    """Create ``dir_path`` and missing parents."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def clr(adata: AnnData, inplace: bool = True, axis: int = 0):
    """Apply centered log-ratio normalization to ``adata.X``.

    Parameters
    ----------
    adata
        AnnData object containing non-negative protein counts.
    inplace
        Modify ``adata`` when true; otherwise return a normalized copy.
    axis
        ``0`` normalizes features and ``1`` normalizes cells, matching the
        historical scMMT API.
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if not inplace:
        adata = adata.copy()

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warnings.warn("Converting sparse adata.X to CSC format for axis=0", stacklevel=2)
        values = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warnings.warn("Converting sparse adata.X to CSR format for axis=1", stacklevel=2)
        values = csr_matrix(adata.X)
    else:
        values = adata.X.copy()

    if values.shape[axis] == 0:
        raise ValueError("Cannot CLR-normalize an empty matrix axis")
    stored_values = values.data if issparse(values) else np.asarray(values)
    if not np.isfinite(stored_values).all() or np.any(stored_values < 0):
        raise ValueError("CLR normalization requires finite, non-negative values")
    if issparse(values):
        values.data /= np.repeat(
            np.exp(np.log1p(values).sum(axis=axis).A / values.shape[axis]),
            values.getnnz(axis=axis),
        )
        np.log1p(values.data, out=values.data)
    else:
        np.log1p(
            values / np.exp(np.log1p(values).sum(axis=axis, keepdims=True) / values.shape[axis]),
            out=values,
        )

    adata.X = values
    return None if inplace else adata


def make_dense(adata: AnnData) -> None:
    """Convert ``adata.X`` to a dense NumPy array when needed."""
    if issparse(adata.X):
        adata.X = adata.X.toarray()
