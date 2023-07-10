from numpy import setdiff1d, intersect1d
from scipy.sparse import issparse
from os.path import isfile
from torch.nn.functional import cross_entropy
from torch.cuda import is_available
from torch.nn import MSELoss
from torch import load as torch_load, save,tensor

from .Utils import build_dir
from .Preprocessing import preprocess
from .Data_Infrastructure.DataLoader_Constructor import build_dataloaders
from .Network.Model import scMMT_Model
from .Network.Losses import no_loss,mse_loss

from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class scMMT_API(object):
    def __init__(self, gene_trainsets, protein_trainsets=[], gene_test = None, gene_list = [], select_hvg = True, train_batchkeys = None, test_batchkey = None, type_key = None, cell_normalize = True, log_normalize = True, gene_normalize = True, min_cells = 30, min_genes = 200, n_svd = 300, n_fa=64, n_hvg=1000,dataset_batch = True,data_dir = None,data_load=False,add_meta=[],batch_size = 128, val_split = "by_test",val_frac=0.1, use_gpu = True,seed=5,log_weight=None):                
        
        if use_gpu:
            print("Searching for GPU")
            
            if is_available():
                print("GPU detected, using GPU")
                self.device = 'cuda'
                
            else:
                print("GPU not detected, falling back to CPU")
                self.device = 'cpu'
                
        else:
            print("Using CPU")
            self.device = 'cpu'
            
        preprocess_args = (gene_trainsets, protein_trainsets, gene_test, train_batchkeys, test_batchkey, type_key,
                     gene_list, select_hvg, cell_normalize, log_normalize, gene_normalize, min_cells,                                                  min_genes,n_svd, n_fa, n_hvg,dataset_batch,data_dir,data_load,seed)

        genes, proteins, genes_test, bools, train_keys, categories = preprocess(*preprocess_args)
        
        if len(add_meta)>0:   
            genes.obsm['result'] = np.concatenate([genes.obsm['result'],genes.obs[add_meta]],axis=1)
            genes_test.obsm['result'] = np.concatenate([genes_test.obsm['result'],genes_test.obs[add_meta]],axis=1)        
        
        self.proteins = proteins
        self.gene = genes.obsm['result']
        self.train_cells = genes.obs.copy()
        self.type_key = type_key
        self.categories = categories
        
        if log_weight == "no_log":
            label = genes.obs[type_key].replace(categories).values
            self.weight = tensor(compute_class_weight(class_weight='balanced', classes=list(categories.values()), y=label),dtype=torch.float32, device = self.device)
        elif log_weight is not None:
            label = genes.obs[type_key].replace(categories).values
            self.weight = tensor(np.log(compute_class_weight(class_weight='balanced', classes=list(categories.values()), y=label)+log_weight) ,dtype=torch.float32, device = self.device)
        else:
            self.weight = None
        
        if genes_test is not None:
            self.test_cells = genes_test.obs.copy()
        else:
            self.test_cells = None
            
        if categories is not None:
            celltypes = proteins.obs[type_key]
        else:
            celltypes = None
        
        if val_split == "by_test":
            print("PCA")
            pca = PCA(n_components = 100)
            train = pca.fit_transform(genes.X)
            test = pca.transform(genes_test.X) 
            print("KNN")
            neigh = NearestNeighbors(n_neighbors=1,metric="cosine")
            neigh.fit(train)
            temp = neigh.kneighbors(test,return_distance=False)
            temp = list(set(temp.reshape(1,-1)[0]))
            val_split = temp
        
        dataloaders = build_dataloaders(genes, proteins, genes_test, bools,train_keys,val_split,val_frac,batch_size,self.device,celltypes,categories)
        self.dataloaders = {key: loader for key, loader in zip(['train', 'val', 'impute', 'test'], dataloaders)}
        
    def train(self,n_epochs = 10000, ES_max = 12, decay_max = 6, h_size=512,drop_rate=0.25,n_layer=4,label_smoothing=0.01, 
              decay_step = 0.1, lr = 10**(-3), weights_dir = None,load = False):
            
        protein_mloss = mse_loss()
        
        if self.categories is not None:
            type_loss = cross_entropy
        else:
            type_loss = no_loss(self.device)
        
        p_mod1, p_mod2 = self.gene.shape[1], self.proteins.obsm["result"].shape[1]
        model_params = {'p_mod1': p_mod1, 'p_mod2': p_mod2, 'loss1': type_loss, 'loss2': protein_mloss, 'categories': self.categories, 'weight':self.weight, "h_size":h_size, "drop_rate":drop_rate, "n_layer":n_layer, "label_smoothing":label_smoothing}

        self.model = scMMT_Model(**model_params)
        self.model.to(self.device)
        
#         build_dir(weights_dir)
        path = weights_dir
        if load and isfile(path):
            self.model.load_state_dict(torch_load(path))
            
        else:
            train_params = (self.dataloaders['train'], self.dataloaders['val'], n_epochs, ES_max, decay_max, decay_step, lr, path,self.device)
        
            self.model.train_backprop(*train_params)
            self.model.load_state_dict(torch_load(path))
#             save(self.model.state_dict(), path_new)
        
    def impute(self,):
        return self.model.impute(self.dataloaders['impute'], self.proteins)
        
    def predict(self,):
        assert self.test_cells is not None
        return self.model.predict(self.dataloaders['test'], self.proteins, self.test_cells)
    
    def embed(self,):
        if self.test_cells is not None:
            loaders = self.dataloaders['impute'], self.dataloaders['test']
        else:
            loaders = self.dataloaders['impute'], None
        return self.model.embed(*loaders, self.train_cells, self.test_cells)
