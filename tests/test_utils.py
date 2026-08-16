import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scMMT import clr


def test_sparse_and_dense_clr_agree_for_cells():
    values = np.array([[1.0, 0.0, 3.0], [2.0, 4.0, 0.0]])
    dense = clr(AnnData(values.copy()), inplace=False, axis=1)
    sparse = clr(AnnData(csr_matrix(values)), inplace=False, axis=1)

    assert np.allclose(dense.X, sparse.X.toarray())


def test_clr_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        clr(AnnData(np.array([[1.0, -1.0]])))
