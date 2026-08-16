"""Public high-level API for scMMT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda import is_available
from torch.nn.functional import cross_entropy

from .Data_Infrastructure.DataLoader_Constructor import build_dataloaders
from .Network.Losses import mse_loss, no_loss
from .Network.Model import scMMT_Model
from .Preprocessing import preprocess


class scMMT_API:
    """Preprocess data, train scMMT, and expose its three output tasks.

    Parameters are intentionally compatible with the original 1.0 API.  RNA
    and protein references must be lists of aligned ``AnnData`` objects.  Set
    ``protein_trainsets=None`` for the RNA-only cell-annotation variant.
    """

    def __init__(
        self,
        gene_trainsets,
        protein_trainsets=None,
        gene_test=None,
        gene_list=None,
        select_hvg=True,
        train_batchkeys=None,
        test_batchkey=None,
        type_key=None,
        cell_normalize=True,
        log_normalize=True,
        gene_normalize=True,
        min_cells=30,
        min_genes=200,
        n_svd=300,
        n_fa=64,
        n_hvg=1000,
        dataset_batch=True,
        data_dir=None,
        data_load=False,
        add_meta=None,
        batch_size=128,
        val_split="by_test",
        val_frac=0.1,
        use_gpu=True,
        seed=5,
        log_weight=None,
    ):
        if seed < 0:
            raise ValueError("seed cannot be negative")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if use_gpu and is_available():
            print("GPU detected, using GPU")
            self.device = "cuda"
        else:
            if use_gpu:
                print("GPU not detected, falling back to CPU")
            else:
                print("Using CPU")
            self.device = "cpu"

        genes, proteins, genes_test, bools, train_keys, categories = preprocess(
            gene_trainsets,
            protein_trainsets,
            gene_test,
            train_batchkeys,
            test_batchkey,
            type_key,
            gene_list,
            select_hvg,
            cell_normalize,
            log_normalize,
            gene_normalize,
            min_cells,
            min_genes,
            n_svd,
            n_fa,
            n_hvg,
            dataset_batch,
            data_dir,
            data_load,
            seed,
        )

        metadata_columns = list(add_meta or [])
        if metadata_columns:
            missing_train = set(metadata_columns) - set(genes.obs.columns)
            if missing_train:
                raise KeyError(f"Training metadata columns not found: {sorted(missing_train)}")
            genes.obsm["result"] = np.concatenate(
                [genes.obsm["result"], genes.obs[metadata_columns].to_numpy()], axis=1
            ).astype(np.float32)
            if genes_test is not None:
                missing_test = set(metadata_columns) - set(genes_test.obs.columns)
                if missing_test:
                    raise KeyError(f"Query metadata columns not found: {sorted(missing_test)}")
                genes_test.obsm["result"] = np.concatenate(
                    [
                        genes_test.obsm["result"],
                        genes_test.obs[metadata_columns].to_numpy(),
                    ],
                    axis=1,
                ).astype(np.float32)

        self.proteins = proteins
        self.gene = genes.obsm["result"]
        self.train_cells = genes.obs.copy()
        self.type_key = type_key
        self.categories = categories

        if log_weight is not None:
            if categories is None or type_key is None:
                raise ValueError("log_weight requires type_key")
            labels = genes.obs[type_key].map(categories).to_numpy()
            balanced = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(len(categories)),
                y=labels,
            )
            if log_weight == "no_log":
                values = balanced
            else:
                shifted = balanced + log_weight
                if np.any(shifted <= 1):
                    raise ValueError(
                        "log_weight must make every logarithmic class weight positive; "
                        "use 'no_log' for ordinary balanced weights"
                    )
                values = np.log(shifted)
            self.weight = torch.tensor(values, dtype=torch.float32, device=self.device)
        else:
            self.weight = None

        self.test_cells = None if genes_test is None else genes_test.obs.copy()
        celltypes = proteins.obs[type_key] if categories is not None else None

        if val_split == "by_test":
            if genes_test is None:
                raise ValueError(
                    "val_split='by_test' requires gene_test; use val_split=None "
                    "for a random validation split"
                )
            n_components = min(100, genes.n_obs - 1, genes.n_vars)
            if n_components < 1:
                raise ValueError("Not enough data to construct a validation split")
            print("Selecting validation cells by query-neighbour similarity")
            pca = PCA(n_components=n_components, random_state=seed)
            train_embedding = pca.fit_transform(np.asarray(genes.X))
            test_embedding = pca.transform(np.asarray(genes_test.X))
            neighbours = NearestNeighbors(n_neighbors=1, metric="cosine")
            neighbours.fit(train_embedding)
            val_split = np.unique(
                neighbours.kneighbors(test_embedding, return_distance=False).ravel()
            ).tolist()

        dataloaders = build_dataloaders(
            genes,
            proteins,
            genes_test,
            bools,
            train_keys,
            val_split,
            val_frac,
            batch_size,
            self.device,
            celltypes,
            categories,
            seed,
        )
        self.dataloaders = dict(zip(("train", "val", "impute", "test"), dataloaders))
        self.model = None

    def train(
        self,
        n_epochs=10000,
        ES_max=12,
        decay_max=6,
        h_size=512,
        drop_rate=0.25,
        n_layer=4,
        label_smoothing=0.01,
        decay_step=0.1,
        lr=1e-3,
        weights_dir=None,
        load=False,
    ):
        """Train a new model or load a compatible state dictionary."""

        type_loss = cross_entropy if self.categories is not None else no_loss(self.device)
        self.model = scMMT_Model(
            p_mod1=self.gene.shape[1],
            p_mod2=self.proteins.obsm["result"].shape[1],
            loss1=type_loss,
            loss2=mse_loss(),
            categories=self.categories,
            weight=self.weight,
            h_size=h_size,
            drop_rate=drop_rate,
            n_layer=n_layer,
            label_smoothing=label_smoothing,
        ).to(self.device)

        if load:
            if weights_dir is None or not Path(weights_dir).is_file():
                raise FileNotFoundError("weights_dir must point to an existing checkpoint")
            try:
                state = torch.load(weights_dir, map_location=self.device, weights_only=True)
            except TypeError:  # PyTorch versions before weights_only was added
                state = torch.load(weights_dir, map_location=self.device)
            self.model.load_state_dict(state)
            return None

        return self.model.train_backprop(
            self.dataloaders["train"],
            self.dataloaders["val"],
            n_epochs,
            ES_max,
            decay_max,
            decay_step,
            lr,
            weights_dir,
            self.device,
        )

    def _require_model(self):
        if self.model is None:
            raise RuntimeError("Call train() before requesting model outputs")

    def impute(self):
        self._require_model()
        return self.model.impute(self.dataloaders["impute"], self.proteins)

    def predict(self):
        self._require_model()
        if self.test_cells is None:
            raise RuntimeError("predict() requires gene_test")
        return self.model.predict(self.dataloaders["test"], self.proteins, self.test_cells)

    def embed(self):
        self._require_model()
        test_loader = self.dataloaders["test"] if self.test_cells is not None else None
        return self.model.embed(
            self.dataloaders["impute"],
            test_loader,
            self.train_cells,
            self.test_cells,
        )
