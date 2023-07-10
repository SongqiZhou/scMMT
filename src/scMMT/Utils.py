from os.path import split, basename, isdir
from os import mkdir
from anndata import AnnData
from scipy.sparse import issparse, csr_matrix,csc_matrix
format_loss = lambda loss, nbatch: round(loss/nbatch, 2)
import numpy as np
def build_dir(dir_path):
    """ This function builds a directory if it does not exist.
    
    
    Arguments:
    ------------------------------------------------------------------
    - dir_path: `str`, The directory to build. E.g. if dir_path = 'folder1/folder2/folder3', then this function will creates directory if folder1 if it does not already exist. Then it creates folder1/folder2 if folder2 does not exist in folder1. Then it creates folder1/folder2/folder3 if folder3 does not exist in folder2.
    """
    
    subdirs = [dir_path]
    substring = dir_path

    while substring != '':
        splt_dir = split(substring)
        substring = splt_dir[0]
        subdirs.append(substring)
        
    subdirs.pop()
    subdirs = [x for x in subdirs if basename(x) != '..']
    
    for dir_ in subdirs[::-1]:
        if not isdir(dir_):
            mkdir(dir_)

            
def clr(adata=AnnData, inplace= True, axis = 0):
    """
    Apply the centered log ratio (CLR) transformation
    to normalize counts in adata.X.
    Args:
        data: AnnData object with protein expression counts.
        inplace: Whether to update adata.X inplace.
        axis: Axis across which CLR is performed.
    """

    if axis not in [0, 1]:
        raise ValueError("Invalid value for `axis` provided. Admissible options are `0` and `1`.")

    if not inplace:
        adata = adata.copy()

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warn("adata.X is sparse but not in CSC format. Converting to CSC.")
        x = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warn("adata.X is sparse but not in CSR format. Converting to CSR.")
        x = csr_matrix(adata.X)
    else:
        x = adata.X
        
    if issparse(x):
        
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]), x.getnnz(axis=axis)
        )
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis]),
            out=x,
        )

    adata.X = x

    return None if inplace else adata 


def make_dense(anndata):
    if issparse(anndata.X):
        tmp = anndata.X.copy()
        anndata.X = tmp.copy().toarray()