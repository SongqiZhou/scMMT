from math import log, exp
from numpy import inf, zeros, zeros_like as np_zeros_like, arange, asarray, empty
from pandas import concat
from anndata import AnnData
import torch
import numpy as np

from torch import cat, no_grad, randn, zeros_like, zeros as torch_zeros, ones, argmax
from torch.nn import Module, Linear, Sequential, RNNCell, Softplus, Parameter, Softmax, SELU,Tanh,L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .Layers import Input_Block, Resnet,Resnet_last


from torch.nn import Module, Linear, BatchNorm1d, PReLU, Dropout, GRU ,AlphaDropout, Tanh
class scMMT_Model(Module):
    def __init__(self, p_mod1, p_mod2, loss1, loss2, categories, weight,h_size, drop_rate,n_layer,label_smoothing):
        super(scMMT_Model, self).__init__()
        self.p_mod2 = p_mod2
        h_size, drop_rate = 512, 0.25
        
        self.input_block = Input_Block(p_mod1, h_size, drop_rate, drop_rate)       
        self.resnet = Sequential(*[Resnet(h_size) for i in range(n_layer)])        
        self.resnet_last = Resnet_last(h_size)
        MSE_output = Sequential(Linear(h_size, p_mod2))
        
        self.mod2_out = MSE_output
                
        if categories is not None:
            self.celltype_out = Sequential(Linear(h_size, len(categories)))
            self.forward = self.forward_transfer
            self.categories_arr = empty((len(categories), ), dtype = 'object')
            for cat in categories:
                self.categories_arr[categories[cat]] = cat
        else:
            self.forward = self.forward_simple
            self.categories_arr = None
            
        self.loss1, self.loss2 = loss1, loss2
        self.label_smoothing = label_smoothing
        self.weight = weight
        
    def forward_transfer(self, x):
        
        x = x.to(torch.float32)
        x = self.input_block(x)
        h = self.resnet(x)
        h = self.resnet_last(h)
        return {'celltypes': self.celltype_out(h), 'modality 2': self.mod2_out(h), 'embedding': h}
 
    def forward_simple(self, x):
        
        x = x.to(torch.float32)
        x = self.input_block(x)
        h = self.resnet(x)
        h = self.resnet_last(h)
        return {'celltypes': None, 'modality 2': self.mod2_out(h), 'embedding': h}
    
    def train_backprop(self, train_loader, val_loader,
                 n_epoch = 10000, ES_max = 30, decay_max = 10, decay_step = 0.1, lr = 10**(-3), path=None, device="cpu"):
        alph = torch.Tensor([0.15]).to(device)
        ES_max = torch.Tensor([ES_max]).to(device)
        if self.categories_arr is None:
            optimizer = Adam(self.parameters(), lr = lr)
            scheduler = StepLR(optimizer, step_size = 1, gamma = decay_step)
            patience = torch.zeros(1).to(device)
            bestloss = torch.Tensor([np.inf]).to(device)
            get_correct = lambda x: 0
            for epoch in range(n_epoch):
                self.train()
                for batch, inputs in enumerate(train_loader):
                    optimizer.zero_grad()
                    mod1, mod2, protein_bools, celltypes = inputs    
                    outputs = self(mod1)
                    mod2_loss = self.loss2(outputs['modality 2'], mod2, protein_bools)
                    loss = mod2_loss
                    loss.backward()
                    optimizer.step()
                with no_grad():
                    running_loss, rtype_acc = torch.zeros(1).to(device), torch.zeros(1).to(device)
                    self.eval()
                    for batch, inputs in enumerate(val_loader):
                        mod1, mod2, protein_bools, celltypes = inputs
                        outputs = self(mod1)
                        n_correct = get_correct(outputs)
                        mod2_loss = self.loss2(outputs['modality 2'], mod2, protein_bools)
                        rtype_acc += n_correct
                        running_loss += mod2_loss.item() * len(mod2)
                    print(f"Epoch {epoch} prediction loss = {running_loss.item()/len(val_loader):.3f}")
                    patience += 1
                    if bestloss > running_loss:
                        bestloss, patience = running_loss, 0
                        torch.save(self.state_dict(),path)
                    if (patience + 1) % decay_max == 0:
                        scheduler.step()
                        print(f"Decaying loss to {optimizer.param_groups[0]['lr']}")
                    if (patience + 1) > ES_max:
                        break
      
        else:
            get_correct = lambda outputs: (argmax(outputs['celltypes'], axis = 1) == celltypes).sum()        
            Weightloss1 = torch.FloatTensor([1]).to(device).requires_grad_(True)
            Weightloss2 = torch.FloatTensor([1]).to(device).requires_grad_(True)
            params = [Weightloss1, Weightloss2]
            optimizer1 = torch.optim.Adam(self.parameters(), lr=lr)
            optimizer2 = torch.optim.Adam(params, lr=lr)
            scheduler1 = StepLR(optimizer1, step_size = 1, gamma = decay_step)
            scheduler2 = StepLR(optimizer2, step_size = 1, gamma = decay_step)      
            patience = torch.zeros(1).to(device)
            bestloss = torch.Tensor([np.inf]).to(device)
            bestloss2 = torch.Tensor([np.inf]).to(device)
            bestacc = torch.zeros(1).to(device)
            Gradloss = L1Loss()        
            for epoch in range(n_epoch):
                self.train()
                for batch, inputs in enumerate(train_loader):
                    mod1, mod2, protein_bools, celltypes = inputs    
                    outputs = self(mod1)
                    mod1_loss = params[0]*self.loss1(outputs['celltypes'], celltypes, weight = self.weight,label_smoothing=self.label_smoothing)
                    mod2_loss = params[1]*self.loss2(outputs['modality 2'], mod2, protein_bools)
                    loss = torch.div(mod1_loss+mod2_loss, 2)
                    if epoch == 0:
                        l01 = mod1_loss.data  
                        l02 = mod2_loss.data
                    optimizer1.zero_grad()
                    loss.backward(retain_graph=True) 
                    param = list(self.parameters())
                    G1R = torch.autograd.grad(mod1_loss, param[0], retain_graph=True, create_graph=True)
                    G1 = torch.norm(G1R[0], 2)
                    G2R = torch.autograd.grad(mod2_loss, param[0], retain_graph=True, create_graph=True)
                    G2 = torch.norm(G2R[0], 2)
                    G_avg = torch.div(G1+G2, 2)
                    lhat1 = torch.div(mod1_loss,l01)
                    lhat2 = torch.div(mod2_loss,l02)
                    lhat_avg = torch.div(lhat1+lhat2, 2)
                    inv_rate1 = torch.div(lhat1,lhat_avg)
                    inv_rate2 = torch.div(lhat2,lhat_avg)
                    C1 = G_avg*(inv_rate1)**alph
                    C2 = G_avg*(inv_rate2)**alph
                    C1 = C1.detach()
                    C2 = C2.detach()
                    optimizer2.zero_grad()
                    Lgrad = Gradloss(G1, C1)+Gradloss(G2, C2)
                    Lgrad.backward()
                    optimizer2.step()
                    optimizer1.step()
                    coef = 2/(Weightloss1+Weightloss2)
                    params = [coef*Weightloss1, coef*Weightloss2]
                with no_grad():
                    running_loss,rtype_acc =torch.zeros(1).to(device),torch.zeros(1).to(device)
                    self.eval()
                    for batch, inputs in enumerate(val_loader):
                        mod1, mod2, protein_bools, celltypes = inputs
                        outputs = self(mod1)
                        n_correct = get_correct(outputs)
                        mod4_loss = self.loss2(outputs['modality 2'], mod2, protein_bools)
                        rtype_acc += n_correct
                        running_loss += mod4_loss.item() * len(mod2)
                        
                    print(f"Epoch {epoch} validation accuracy = {rtype_acc.item()/len(val_loader):.3f},mseloss={running_loss.item()/len(val_loader):.3f}")
                    patience += 1                              
                    if ((bestloss / (running_loss/len(val_loader))).item() +\
                        ((rtype_acc/len(val_loader))/bestacc)).item()>2:
                        bestloss,bestacc, patience = running_loss/len(val_loader), rtype_acc/len(val_loader), torch.zeros(1).to(device)
                        torch.save(self.state_dict(),path)
                    if self.p_mod2 == 0:
                        if ((rtype_acc/len(val_loader))/bestacc).item()>1:
                            bestacc, patience = rtype_acc/len(val_loader), torch.zeros(1).to(device)
                            torch.save(self.state_dict(),path)
                        
                    if (patience + 1) % decay_max == 0:
                        scheduler1.step()
                        scheduler2.step()
                        print(f"Decaying loss to {optimizer1.param_groups[0]['lr']}")
                    if (patience + 1) > ES_max:
                        break
    
    def impute(self, impute_loader, proteins):
        imputed_test = proteins.copy()
        self.eval()
        start = 0
        for mod1, bools, celltypes in impute_loader:
            end = start + mod1.shape[0]
            with no_grad():
                outputs = self(mod1)
            mod2_impute = outputs['modality 2']
            imputed_test.X[start:end] = self.fill_predicted(imputed_test.X[start:end], mod2_impute, bools)
            start = end
        return imputed_test
    
    def embed(self, impute_loader, test_loader, cells_train, cells_test):
        if cells_test is not None:
            embedding = AnnData(zeros(shape = (len(cells_train) + len(cells_test), 512)))
            embedding.obs = concat((cells_train, cells_test), join = 'inner')
        else:
            embedding = AnnData(zeros(shape = (len(cells_train), 512)))
            embedding.obs = cells_train
        self.eval()
        start = 0
        for mod1, bools, celltypes in impute_loader:
            end = start + mod1.shape[0]
            outputs = self(mod1)
            embedding[start:end] = outputs['embedding'].detach().cpu().numpy()
            start = end
        if cells_test is not None:
            for mod1 in test_loader:
                end = start + mod1.shape[0]
                outputs = self(mod1)
                embedding[start:end] = outputs['embedding'].detach().cpu().numpy()
                start = end
        return embedding
                    
    def fill_predicted(self, array, predicted, bools):
        bools = bools.cpu().numpy()
        return (1. - bools) * predicted.cpu().numpy() + array
    
    def predict(self, test_loader, proteins, cells):
        imputed_test = AnnData(zeros(shape = (len(cells), self.p_mod2)),dtype=np.float32)
        imputed_test.obs = cells
        imputed_test.var.index = proteins.var.index
        if self.categories_arr is not None:
            celltypes = ['None'] * len(cells)
        self.eval()
        start = 0
        for mod1 in test_loader:
            end = start + mod1.shape[0]
            with no_grad():
                outputs = self(mod1)
            if self.categories_arr is not None:
                predicted_types = argmax(outputs['celltypes'], axis = 1).cpu().numpy()
                celltypes[start:end] = self.categories_arr[predicted_types].tolist()
            mod2_impute = outputs['modality 2']
            imputed_test[start:end] = mod2_impute.cpu().numpy()
            start = end
        if self.categories_arr is not None:
            imputed_test.obs['transfered cell labels'] = celltypes
        return imputed_test