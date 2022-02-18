

class Config(object):
        
    def _parse(self, kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        #
        
        print('=================================')
        print('*', self.config_name)
        print('---------------------------------')
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))
        print('=================================')
        
        
    def __str__(self,):
        
        buff = ""
        buff += '=================================\n'
        buff += ('*'+self.config_name+'\n')
        buff += '---------------------------------\n'
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                buff += ('-' + str(k) + ':' + str(getattr(self, k))+'\n')
            #
        #
        buff += '=================================\n'
        
        return buff    
    

class AjointMAML_Config(Config):
    
    domain  = None
    n_way   = None     
    k_shot  = None     
    k_query = None     
    k_test  = 1000     
    
    tr_tasks = 100
    te_tasks = 100
    
    batchsize = 10000
    
    inner_n_steps = None
    inner_stepsize = None
    
    meta_batch = 5
    meta_shuffle_batch=False
    meta_lr = 0.1
    meta_reg = 1.0
    meta_epochs = 5
    
    imaml_reg = False
    heun = True
    
    base_module = None
    
    imgsz = 84
    imgc = 3
    
    # run device config
    device='cuda:0'
    dtype='float64'
    
    verbose=True
    
    def __init__(self,):
        super(AjointMAML_Config, self).__init__()
        self.config_name = 'Ajoint-MAML-Config'
        
class VanillaMAML_Config(Config):
    
    domain  = None
    n_way   = None     
    k_shot  = None     
    k_query = None     
    k_test  = 1000     
    
    tr_tasks = 100
    te_tasks = 100
    
    batchsize = 10000
    
    inner_n_steps = None
    inner_stepsize = None
    
    meta_batch = 5
    meta_shuffle_batch=False
    meta_lr = 0.1
    meta_reg = 1.0
    meta_epochs = 5
    imaml_reg = False
    
    base_module = None
    
    imgsz = 84
    imgc = 3
    
    # run device config
    device='cuda:0'
    dtype='float64'
    
    verbose=True
    
    def __init__(self,):
        super(VanillaMAML_Config, self).__init__()
        self.config_name = 'Vanilla-MAML-Config'
        
class ImplicitMAML_Config(Config):
    
    domain  = None
    n_way   = None     
    k_shot  = None     
    k_query = None     
    k_test  = 1000     
    
    tr_tasks = 100
    te_tasks = 100
    
    batchsize = 10000
    
    inner_n_steps = None
    inner_stepsize = None
    
    meta_batch = 5
    meta_shuffle_batch=False
    meta_lr = 0.1
    meta_reg = 1.0
    meta_epochs = 5
    imaml_reg = False
    
    base_module = None
    
    imgsz = 84
    imgc = 3
    
    # run device config
    device='cuda:0'
    dtype='float64'
    
    verbose=True
    
    ## unique for implicit MAML
    cg_steps = 5
    cg_damping = 1.0
    lam_lr = 0.0
    lam_min = 0.0
    scalar_lam = True
    taylor_approx = False
    inner_alg = 'gradient'
    outer_alg = 'gradient'
    
    def __init__(self,):
        super(ImplicitMAML_Config, self).__init__()
        self.config_name = 'Implicit-MAML-Config'
        
        