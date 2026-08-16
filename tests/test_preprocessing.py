import numpy as np
from anndata import AnnData

from scMMT.Preprocessing import preprocess


def _rna(rows, genes=("g1", "g2", "g3", "g4")):
    data = AnnData(np.arange(1, rows * len(genes) + 1, dtype=np.float32).reshape(rows, -1))
    data.var_names = list(genes)
    data.obs_names = [f"cell-{index}" for index in range(rows)]
    data.obs["celltype"] = ["a", "b"] * (rows // 2)
    return data


def test_preprocessing_without_gene_scaling_still_builds_protein_panel():
    genes = _rna(4)
    proteins = AnnData(np.arange(8, dtype=np.float32).reshape(4, 2), obs=genes.obs.copy())
    proteins.var_names = ["CD4", "CD8"]

    result = preprocess(
        [genes],
        [proteins],
        type_key="celltype",
        select_hvg=False,
        cell_normalize=False,
        log_normalize=False,
        gene_normalize=False,
        min_cells=0,
        min_genes=0,
        n_svd=2,
        n_fa=2,
        data_dir=None,
    )

    gene_train, protein_train, gene_test, masks, train_keys, categories = result
    assert gene_test is None
    assert protein_train.obsm["result"].shape == (4, 2)
    assert masks.tolist() == [[1.0, 1.0]]
    assert train_keys == []
    assert categories == {"a": 0, "b": 1}
    assert gene_train.obsm["result"].shape[0] == 4


def test_categories_are_computed_after_cell_qc():
    genes = _rna(4)
    genes.X[0] = 0
    genes.obs["celltype"] = ["removed", "kept", "kept", "kept"]
    proteins = AnnData(np.ones((4, 1), dtype=np.float32), obs=genes.obs.copy())
    proteins.var_names = ["CD4"]

    *_, categories = preprocess(
        [genes],
        [proteins],
        type_key="celltype",
        select_hvg=False,
        cell_normalize=False,
        log_normalize=False,
        gene_normalize=False,
        min_cells=0,
        min_genes=1,
        n_svd=2,
        n_fa=2,
        data_dir=None,
    )

    assert categories == {"kept": 0}
