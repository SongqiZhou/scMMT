from torch import cat
from torch.nn import Module, Linear, BatchNorm1d, PReLU, Dropout,SELU,Tanh,GELU, RNNCell
import torch

class Input_Block(Module):
    def __init__(self, in_units, out_units, dropout_inrate, dropout_outrate):
        super(Input_Block, self).__init__()
        self.bnorm_in = BatchNorm1d(in_units)
        self.dropout_in = Dropout(dropout_inrate)
        self.dense = Linear(in_units, out_units)
        self.act = SELU()
            
    def forward(self, x_new):
        x_new = self.bnorm_in(x_new)
        x_new = self.dropout_in(x_new)
        x = self.dense(x_new)
        h = self.act(x)
        return h
    
class Resnet(Module):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super(Resnet, self).__init__()
        self.Dropout = Dropout(dropout_rate)
        self.dense = Linear(hidden_units, hidden_units)
        self.act =  SELU()
        
    def forward(self, x):
        h = self.Dropout(x)
        h = self.dense(h)
        h = self.act(h)
        return x+h

class Resnet_last(Module):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super(Resnet_last, self).__init__()
        self.Dropout = Dropout(dropout_rate)
        self.dense = Linear(hidden_units, hidden_units)
        self.act =  SELU()
        
    def forward(self, x):
        x = self.Dropout(x)
        x = self.dense(x)
        x = self.act(x)
        return x