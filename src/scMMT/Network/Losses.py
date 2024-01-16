from torch import tensor, abs as torch_abs, logical_not, log, clamp
from torch.nn import Softplus
import torch

class no_loss(object):
    def __init__(self, device):
        self.device = device
    
    def __call__(self, outputs, target):
        return tensor(0., device = self.device)
    
    
class mse_loss(object):
    def __init__(self, reduce = True):
        self.reduce = reduce
        
    def __call__(self, yhat, y, bools=None):
        SEs = (yhat - y)**2
        
        if self.reduce:
            if bools != None:
                loss = SEs * bools
                return loss.mean()
            return SEs.mean()    
        return SEs
