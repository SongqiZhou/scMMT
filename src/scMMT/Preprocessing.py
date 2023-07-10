from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
from tqdm import tqdm
from time import sleep
import numpy as np
from numpy import intersect1d, setdiff1d, quantile, unique, asarray, zeros
from numpy.random import choice, seed
from .Utils import make_dense
import pickle
from anndata import AnnData
import scanpy as sc
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD, FactorAnalysis

sc.settings.verbosity = 0

def preprocess(gene_trainsets, protein_trainsets=[], gene_test = None, train_batchkeys = None, test_batchkey = None, type_key = None, gene_list = [], select_hvg = True, cell_normalize = True, log_normalize = True, gene_normalize = True, min_cells = 1, min_genes = 1,n_svd = 300, n_fa=180, n_hvg=550,dataset_batch = True,data_dir="data.pkl",data_load=False,seed=5):
    
    assert type(gene_trainsets) == list
    assert type(protein_trainsets) == list
    assert all([sum(x.obs.index == y.obs.index)/len(x) == 1. for x, y in zip(gene_trainsets, protein_trainsets)])
    
    if protein_trainsets==[]:
        for tmp in gene_trainsets:
            protein_tmp = AnnData(np.zeros(shape = (len(tmp.obs), 0)),obs=tmp.obs,dtype=np.int32)
            protein_trainsets.append(protein_tmp)    
            
    seed = seed
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    
    if data_dir is not None and data_load:
        #print(data_dir,data_load)
        data = pickle.load(open(data_dir, 'rb'))
        gene_train, protein_train, gene_test, bools, train_keys, categories = data
        return gene_train, protein_train, gene_test, bools, train_keys, categories
    
    if type_key:
        assert all([type_key in x.obs.columns for x in protein_trainsets])
        categories, i = {}, 0
        
        for dataset in protein_trainsets:
            for celltype in dataset.obs[type_key]:
                if celltype not in categories:
                    categories[celltype] = i
                    i += 1
     
    else:
        categories = None
            
    if train_batchkeys is not None:
        assert type(train_batchkeys) == list
        for i in range(len(gene_trainsets)):
            key = train_batchkeys[i]
            gene_trainsets[i].obs['batch'] = ['DS-' + str(i + 1) + ' ' + x for x in gene_trainsets[i].obs[key]]
            protein_trainsets[i].obs['batch'] = ['DS-' + str(i + 1) + ' ' + x for x in protein_trainsets[i].obs[key]]
    else:
        for i in range(len(gene_trainsets)):
            gene_trainsets[i].obs['batch'] = 'DS-' + str(i + 1)
            protein_trainsets[i].obs['batch'] = 'DS-' + str(i + 1)
        
    if gene_test is not None:
        if test_batchkey is not None:
            gene_test.obs['batch'] = ['DS-Test ' + x for x in gene_test.obs[test_batchkey]]
        else:
            gene_test.obs['batch'] = 'DS-Test'
                    
        
    gene_set = set(gene_list)
    
    if min_genes:
        print("\nQC Filtering Training Cells")
        
        for i in range(len(gene_trainsets)):
            cell_filter = (gene_trainsets[i].X > 10**(-8)).sum(axis = 1) >= min_genes
            gene_trainsets[i] = gene_trainsets[i][cell_filter].copy()
            protein_trainsets[i] = protein_trainsets[i][cell_filter].copy()
        
        if gene_test is not None:
            print("QC Filtering Testing Cells")
            
            cell_filter = (gene_test.X > 10**(-8)).sum(axis = 1) >= min_genes
            gene_test = gene_test[cell_filter].copy()
        
    if min_cells:
        print("\nQC Filtering Training Genes")
        
        for i in range(len(gene_trainsets)):
            bools = (gene_trainsets[i].X > 10**(-8)).sum(axis = 0) >= min_cells            
            genes = gene_trainsets[i].var.index[bools]
            genes = asarray(genes).reshape((-1,))
            features = set(genes)
            features.update(gene_set)
            features = list(features)
            features.sort()

            gene_trainsets[i] = gene_trainsets[i][:, features].copy()

        if gene_test is not None:
            print("QC Filtering Testing Genes")

            bools = (gene_test.X > 10**(-8)).sum(axis = 0) >= min_cells
            genes = gene_test.var.index[bools]
            genes = asarray(genes).reshape((-1,))
            features = set(genes)
            features.update(gene_set)
            features = list(features)
            features.sort()

            gene_test = gene_test[:, features].copy()
    for i in range(len(gene_trainsets)):
        gene_trainsets[i].layers["raw"] = gene_trainsets[i].X.copy()
        protein_trainsets[i].layers["raw"] = protein_trainsets[i].X.copy()
    if gene_test is not None:
        gene_test.layers["raw"] = gene_test.X.copy()
            
    if cell_normalize:
        print("\nNormalizing Training Cells")
        
        [sc.pp.normalize_total(x) for x in gene_trainsets]

        if gene_test is not None:
            print("Normalizing Testing Cells")
            sc.pp.normalize_total(gene_test, key_added = "scale_factor")

    if log_normalize:
        print("\nLog-Normalizing Training Data")
        
        [sc.pp.log1p(x) for x in gene_trainsets]
        
        if gene_test is not None:
            print("Log-Normalizing Testing Data")
            sc.pp.log1p(gene_test)
    
    gene_train = gene_trainsets[0]
    gene_train.obs['Dataset'] = 'Dataset 1'
    protein_trainsets[0].obs['Dataset'] = 'Dataset 1'
    for i in range(1, len(gene_trainsets)):
        gene_trainsets[i].obs['Dataset'] = 'Dataset ' + str(i + 1)
        protein_trainsets[i].obs['Dataset'] = 'Dataset ' + str(i + 1)
        gene_train = gene_train.concatenate(gene_trainsets[i], join = 'inner', batch_key = None)
    
    if gene_test is not None:
        genes = intersect1d(gene_train.var.index, gene_test.var.index)
        gene_train = gene_train[:, genes].copy()
        gene_test = gene_test[:, genes].copy()
              
    make_dense(gene_train)
    [make_dense(x) for x in protein_trainsets]
    if gene_test is not None:
        make_dense(gene_test)

    #demension
    tsvd = TruncatedSVD(n_components = n_svd)
    FA = FactorAnalysis(n_components = n_fa)
    if gene_test is not None:
        temp = np.concatenate((np.array(gene_train.X),np.array(gene_test.X)),axis=0)
        if dataset_batch: 
            tmp = AnnData(temp)
            gene_test.obs["dataset"] = "test"
            gene_train.obs["dataset"] = "train"
            tmp.obs = pd.concat([gene_train.obs,gene_test.obs],axis=0)
            print("\ncombat")
            sc.pp.combat(tmp, key='dataset')
            temp = tmp.X.copy()
                 
        print("\nTSVD...")
        temp_svd = tsvd.fit_transform(temp)
        train_col = gene_train.X.shape[0]
        gene_train.obsm['X_svd'] = temp_svd[:train_col]
        gene_test.obsm['X_svd'] = temp_svd[train_col:]
        
#         gene_train.X = temp[:train_col]
#         gene_test.X = temp[train_col:]
        
        print("\nFA ...")
        temp_fa = FA.fit_transform(temp)
        gene_train.obsm['X_fa'] = temp_fa[:train_col]
        gene_test.obsm['X_fa'] = temp_fa[train_col:]
    else:
        print("\nTSVD Train")
        gene_train.obsm['X_svd'] = tsvd.fit_transform(gene_train.X)
        print("\nFA train")
        gene_train.obsm['X_fa'] = FA.fit_transform(gene_train.X)
       
    
    if select_hvg:
        print("\nFinding HVGs")
        
        if gene_test is not None:
            tmp = gene_train.concatenate(gene_test, batch_key = None).copy()
        else:
            tmp = gene_train.copy()
        
        if not cell_normalize or not log_normalize:
            print("Warning, highly variable gene selection may not be accurate if expression is not cell normalized and log normalized")
            
        if len(tmp) > 10**5:
            idx = choice(range(len(tmp)), 10**5, False)
            tmp = tmp[idx].copy()
            
        sc.pp.highly_variable_genes(tmp, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, 
                              n_bins = 20, subset = False, batch_key = 'batch', n_top_genes = n_hvg)
        hvgs = tmp.var.index[tmp.var['highly_variable']].copy()
        tmp = None
        
        gene_set.update(set(hvgs))
        gene_set = list(gene_set)
        gene_set.sort()
        gene_train = gene_train[:, gene_set].copy()
        if gene_test is not None:
            gene_test = gene_test[:, gene_set].copy()
    
    if gene_normalize:
        patients = unique(gene_train.obs['batch'].values)

        print("\nNormalizing Gene Training Data by Batch")
        sleep(1)

        for patient in patients:
            indices = [x == patient for x in gene_train.obs['batch']]
            sub_adata = gene_train[indices].copy()
            sc.pp.scale(sub_adata)

            gene_train[indices] = sub_adata.X.copy()
                
        if gene_test is not None:
            patients = unique(gene_test.obs['batch'].values)

            print("\nNormalizing Gene Testing Data by Batch")
            
            
            for patient in patients:
                indices = [x == patient for x in gene_test.obs['batch']]
                sub_adata = gene_test[indices].copy()
                sc.pp.scale(sub_adata)

                gene_test[indices] = sub_adata.X.copy()

        train_keys, curr_break, proteins = [], len(protein_trainsets[0]), [set(protein_trainsets[0].var.index)]
        protein_train = protein_trainsets[0].copy()
        for i in range(1, len(protein_trainsets)):
            protein_train = protein_train.concatenate(protein_trainsets[i], join = 'outer', fill_value = 0., 
                                                      batch_key = None)
            
            proteins.append(set(protein_trainsets[i].var.index))
            train_keys.append(curr_break)
            curr_break += len(protein_trainsets[i])
        
        bools = asarray([[int(x in prot_set) for x in protein_train.var.index] for prot_set in proteins])
        
        for i in range(len(bools)):
            protein_train.var['Dataset ' + str(i + 1)] = [bool(x) for x in bools[i]]
        
    gene_train.obsm['result'] = np.concatenate([gene_train.obsm['X_svd'],gene_train.obsm['X_fa'],gene_train.X],axis=1)
    if gene_test is not None:
        gene_test.obsm['result'] = np.concatenate([gene_test.obsm['X_svd'],gene_test.obsm['X_fa'] ,gene_test.X],axis=1)
    
    protein_train.obsm['result'] = protein_train.X
    
    data = (gene_train, protein_train, gene_test, bools, train_keys, categories)
    pickle.dump(data,open(data_dir, 'wb'))
    return gene_train, protein_train, gene_test, bools, train_keys, categories
        