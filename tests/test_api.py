import numpy as np
from anndata import AnnData

from scMMT import scMMT_API


def test_high_level_api_runs_all_three_tasks():
    rng = np.random.default_rng(4)
    reference = AnnData(rng.poisson(3, (10, 6)).astype(np.float32))
    reference.var_names = [f"g{index}" for index in range(6)]
    reference.obs_names = [f"reference-{index}" for index in range(10)]
    reference.obs["celltype"] = ["a", "b"] * 5
    proteins = AnnData(
        rng.normal(size=(10, 3)).astype(np.float32),
        obs=reference.obs.copy(),
    )
    proteins.var_names = ["p1", "p2", "p3"]
    query = AnnData(rng.poisson(3, (3, 6)).astype(np.float32))
    query.var_names = reference.var_names.copy()
    query.obs_names = [f"query-{index}" for index in range(3)]

    api = scMMT_API(
        [reference],
        [proteins],
        gene_test=query,
        type_key="celltype",
        select_hvg=False,
        cell_normalize=False,
        log_normalize=False,
        gene_normalize=False,
        min_cells=0,
        min_genes=0,
        n_svd=2,
        n_fa=2,
        dataset_batch=False,
        data_dir=None,
        batch_size=4,
        val_split=None,
        val_frac=0.2,
        use_gpu=False,
        seed=4,
    )
    history = api.train(
        n_epochs=1,
        ES_max=1,
        h_size=8,
        drop_rate=0,
        n_layer=1,
        label_smoothing=0,
    )

    predictions = api.predict()
    imputed = api.impute()
    embedding = api.embed()

    assert len(history) == 1
    assert predictions.shape == (3, 3)
    assert imputed.shape == (10, 3)
    assert embedding.shape == (13, 8)
    assert predictions.obs["transferred cell labels"].notna().all()
