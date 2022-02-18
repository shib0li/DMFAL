import warnings
import numpy as np
import torch

class DefaultConfig(object):
    
    verbose=False
    
    torch_type = torch.float
    placement = None
    
#     device = torch.device('cuda:0')
    
    ##################
    #  model config  #
    ##################
    
    domain = None
    penalty = None
    
    heuristic = None
    
    M = None
    input_dim_list = None
    output_dim_list = None
    base_dim_list = None
#     hlayers_config = None
    hlayers_w = None
    hlayers_d = None

    trial = None
    
    Nquery = 1
    batch_size = None
    
    T = None

    activation=None

    learning_rate = 1e-4
    reg_strength = 1e-3
    opt_lr = None

    print_freq = 100
    max_epochs = 1000



    
    def _parse(self, kwargs):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('WARNING: options does not include attribute %s' % k)
            setattr(self, k, v)
        
        print('Model Config:')
        print('=======================')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, ':', getattr(self, k))
        print('=======================')
        
opt = DefaultConfig()

    
    
    