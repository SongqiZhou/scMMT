"""Batch samplers used by scMMT's lightweight tensor loaders."""

from __future__ import annotations

from math import ceil
from typing import Iterator, Sequence

import numpy as np


class batchSampler:
    """Yield cell indices together with their source-dataset indices.

    ``train_keys`` contains the cumulative boundaries between concatenated
    reference datasets.  The resulting dataset index selects the matching
    protein-observation mask for every cell in a batch.
    """

    def __init__(
        self,
        indices: Sequence[int],
        train_keys: Sequence[int],
        bsize: int,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        if bsize <= 0:
            raise ValueError("bsize must be a positive integer")

        self.indices = np.asarray(indices, dtype=int)
        self.train_keys = np.asarray(train_keys, dtype=int)
        self.bsize = int(bsize)
        self.shuffle = shuffle
        self.rng = rng or np.random.default_rng()

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        indices = self.rng.permutation(self.indices) if self.shuffle else self.indices

        for start in range(0, len(indices), self.bsize):
            minibatch = indices[start : start + self.bsize]
            # searchsorted returns the source dataset for every concatenated row.
            dataset_indices = np.searchsorted(self.train_keys, minibatch, side="right")
            yield minibatch.tolist(), dataset_indices.tolist()

    def __len__(self) -> int:
        """Return the number of batches, matching the PyTorch convention."""
        return ceil(len(self.indices) / self.bsize)


def build_trainSamplers(
    adata,
    n_train: Sequence[int],
    bsize: int = 128,
    val_split: Sequence[int] | np.ndarray | None = None,
    val_frac: float = 0.1,
    seed: int | None = None,
):
    """Build non-overlapping training and validation samplers.

    ``val_split`` may contain explicit validation row indices.  When it is
    ``None``, a reproducible random fraction is selected.  The historical
    string value ``"by_test"`` must be resolved by :class:`scMMT_API` first.
    """

    n_obs = len(adata)
    if n_obs < 2:
        raise ValueError("At least two reference cells are required")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_obs, dtype=int)
    num_val = min(max(round(val_frac * n_obs), 1), n_obs - 1)

    if isinstance(val_split, str):
        if val_split == "by_test":
            raise ValueError(
                "val_split='by_test' requires gene_test so scMMT_API can "
                "derive nearest-neighbour validation cells"
            )
        raise ValueError(f"Unknown val_split mode: {val_split!r}")

    if val_split is None:
        val_indices = rng.choice(all_indices, num_val, replace=False)
    else:
        val_indices = np.asarray(list(val_split), dtype=int)
        if val_indices.ndim != 1 or len(val_indices) == 0:
            raise ValueError("val_split must contain at least one row index")
        if np.any((val_indices < 0) | (val_indices >= n_obs)):
            raise IndexError("val_split contains an out-of-range row index")
        val_indices = np.unique(val_indices)
        if len(val_indices) >= n_obs:
            raise ValueError("val_split must leave at least one training cell")
        if len(val_indices) > num_val:
            val_indices = rng.choice(val_indices, num_val, replace=False)

    train_indices = np.setdiff1d(all_indices, val_indices, assume_unique=True)
    train_sampler = batchSampler(train_indices, n_train, bsize, shuffle=True, rng=rng)
    val_sampler = batchSampler(val_indices, n_train, bsize)
    return train_sampler, val_sampler


def build_testSampler(adata, train_keys: Sequence[int], bsize: int = 128):
    return batchSampler(np.arange(len(adata), dtype=int), train_keys, bsize)
