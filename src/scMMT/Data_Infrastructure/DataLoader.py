from math import ceil

from torch import as_tensor, float32, long


def tensor_loader(*adatas, protein_boolean, sampler, device, celltypes=None, categories=None):
    if (celltypes is not None) and (categories is not None):
        return tensor_loader_trihead(
            *adatas,
            protein_boolean=protein_boolean,
            sampler=sampler,
            device=device,
            celltypes=celltypes,
            categories=categories,
        )

    return tensor_loader_dualhead(
        *adatas, protein_boolean=protein_boolean, sampler=sampler, device=device
    )


class tensor_loader_dualhead:
    def __init__(self, *adatas, protein_boolean, sampler, device):
        if not adatas:
            raise ValueError("At least one AnnData object is required")
        if not all(adata.shape[0] == adatas[0].shape[0] for adata in adatas):
            raise ValueError("All AnnData objects must contain the same cells")

        self.arrs = [adata.obsm["result"] for adata in adatas]
        #         print(arrs.shape)
        self.bools = protein_boolean

        self.sampler = sampler
        self.device = device

    def __iter__(self):
        for idxs, bool_set in self.sampler:
            arrays = [arr[idxs] for arr in self.arrs]
            arrays = [as_tensor(arr, device=self.device, dtype=float32) for arr in arrays]
            arrays.append(as_tensor(self.bools[bool_set], device=self.device, dtype=float32))
            arrays.append(None)

            yield arrays

    def __len__(self):
        return len(self.sampler)


class tensor_loader_trihead:
    def __init__(self, *adatas, protein_boolean, sampler, device, celltypes, categories):
        if not adatas:
            raise ValueError("At least one AnnData object is required")
        if not all(adata.shape[0] == adatas[0].shape[0] for adata in adatas):
            raise ValueError("All AnnData objects must contain the same cells")

        self.arrs = [adata.obsm["result"] for adata in adatas]
        self.celltypes, self.categories = celltypes, categories
        self.bools = protein_boolean

        self.sampler = sampler
        self.device = device

    def __iter__(self):
        for idxs, bool_set in self.sampler:
            arrays = [arr[idxs] for arr in self.arrs]
            arrays = [as_tensor(arr, device=self.device, dtype=float32) for arr in arrays]
            arrays.append(as_tensor(self.bools[bool_set], device=self.device, dtype=float32))
            arrays.append(
                as_tensor(
                    [self.categories[cat] for cat in self.celltypes.iloc[idxs]],
                    device=self.device,
                    dtype=long,
                )
            )

            yield arrays

    def __len__(self):
        return len(self.sampler)


class tensor_loader_basic:
    def __init__(self, adata, batch_size, device):
        self.arr = adata.obsm[
            "result"
        ].copy()  # np.concatenate((adata.X,adata.obsm['X_svd'],adata.obsm['X_fa']),axis=1).copy()
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        batch = []

        for idx in range(len(self.arr)):
            batch.append(idx)

            if len(batch) == self.batch_size:
                yield as_tensor(self.arr[batch], device=self.device, dtype=float32)
                batch = []

        if batch:
            yield as_tensor(self.arr[batch], device=self.device, dtype=float32)

    def __len__(self):
        return ceil(len(self.arr) / self.batch_size)
