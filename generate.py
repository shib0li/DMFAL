import numpy as np
import os
import time

# import hickle as hkl
# from hdf5storage import loadmat

from scipy.io import savemat

import sobol_seq

import fire
from tqdm.auto import trange 

import dataset_active as dataset

class DatasetConfig(object):

    domain = None
    Ntrain = None
    Ntest = None
    trial = None

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
        
config = DatasetConfig()

def generate_sobol_inputs(N, dim, lb, ub):
    noise = sobol_seq.i4_sobol_generate(dim, N)
    scale = (ub - lb).reshape([1,-1])
    X = noise*scale + lb
    return X

def generate_random_inputs(N, dim, lb, ub, seed):
    rand_state = np.random.get_state()
    try:
        np.random.seed(seed)
        noise = np.random.uniform(0,1,size=[N,dim])
        scale = (ub - lb).reshape([1,-1])
    except:
        print('Errors occured when generating random noise...')
    finally:
        np.random.set_state(rand_state)
    #
    X = noise*scale + lb
    return X
    
def prepare(**kwargs):
    
    config._parse(kwargs)
    
    Mfn = {
        'Poisson3': dataset.Poisson3,
        'Poisson2': dataset.Poisson2,
        'Heat3': dataset.Heat3,
        'Heat2': dataset.Heat2,
        'Burgers':dataset.Burgers,
        'Navier': dataset.Navier,
        'Lbracket': dataset.Lbracket,
    }[config.domain]()
    
    # generate train data with sobol random
    D = {}
    D['Ntrain'] = config.Ntrain
    D['Ntest'] = config.Ntest
    D['trial'] = config.trial
    D['domain'] = config.domain
    D['Nf'] = Mfn.M
    
    X_train_list = []
    y_train_list = []
    y_train_ground_list = []
    
    X_test_list = []
    y_test_list = []
    y_test_ground_list = []
    
    for m in range(Mfn.M):
        X_train = generate_sobol_inputs(config.Ntrain[m], Mfn.dim, Mfn.lb, Mfn.ub)
        X_test = generate_random_inputs(config.Ntest[m], Mfn.dim, Mfn.lb, Mfn.ub, seed=config.trial)
        
        y_train = Mfn.query(X_train, m)
        y_train_ground = Mfn.ground(X_train)
        
        print('m=',m)
        print('shape of y', y_train.shape)
        print('shape of y_ground', y_train_ground.shape)

        y_test = Mfn.query(X_test, m)
        y_test_ground = Mfn.ground(X_test)

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        y_train_ground_list.append(y_train_ground)
        y_test_ground_list.append(y_test_ground)
    #
    D['X_train_list'] = X_train_list
    D['y_train_list'] = y_train_list
    D['y_train_ground_list'] = y_train_ground_list
    
    D['X_test_list'] = X_test_list
    D['y_test_list'] = y_test_list
    D['y_test_ground_list'] = y_test_ground_list
    
    dump_path = os.path.join('data/__processed__',config.domain)
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
       
    dump_fname = config.domain + '_trial' + str(config.trial) + '.h5'

    dataset.dump_to_h5f(D, os.path.join(dump_path, dump_fname), comp=None)
    
    

#     # verify
#     Dl = dataset.load_from_h5f(os.path.join(dump_path, dump_fname))
#     M = Dl['Nf']
#     print(M)
#     for m in range(M):
#         print(np.array_equal(Dl['X_train_list'][m], D['X_train_list'][m]))
#         print(np.array_equal(Dl['y_train_list'][m], D['y_train_list'][m]))
#         print(np.array_equal(Dl['y_train_ground_list'][m], D['y_train_ground_list'][m]))

#         print(np.array_equal(Dl['X_test_list'][m], D['X_test_list'][m]))
#         print(np.array_equal(Dl['y_test_list'][m], D['y_test_list'][m]))
#         print(np.array_equal(Dl['y_test_ground_list'][m], D['y_test_ground_list'][m]))

        
if __name__=='__main__':
    fire.Fire(prepare)

