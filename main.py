# import numpy as np
import os
# import time
# from hdf5storage import savemat

import fire
from tqdm.auto import trange 
import pickle

# from config import opt

from hdf5storage import loadmat
import h5py

# import sys
# import dataset_active as dataset

import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
import time
import datetime

from config import opt
# import model.Utils as utils

# import data.dataset_active as dataset
import dataset_active as dataset
# from model.BaseNet import AdaptiveBaseNet

# from DeepMFNet import DeepMFNet
from model.BatchDMFAL import BatchDMFAL
from model.MutualDMFAL import MutualDMFAL
from model.PdvDMFAL import PdvDMFAL

from model.BaldAL import BaldAL

from infrastructure.misc import *


# import matplotlib.pyplot as plt



# TORCH_TYPE = opt.torch_type
# DEVICE = opt.device

def create_path(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        #
        print("Directory '%s' created successfully" % (path))
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #
    
def dump_pred_to_h5f(hist_pred, h5fname, comp=None):
 
    try:
        hf = h5py.File(h5fname, 'w')
        Npred = len(hist_pred)
        
        group_pred_test = hf.create_group('hist_N_pred')
        
        for n in range(Npred):
            key = 'predict_at_t' + str(n)
            pred = hist_pred[n]
            print(pred.shape)
            group_pred_test.create_dataset(key, data=pred, compression=comp)
        #
    except:
        print('ERROR occurs when WRITE the h5 object...')
    finally:
        hf.close()

def evaluation(**kwargs):
    
    opt._parse(kwargs)
    
    res_path = os.path.join('__results__', opt.domain)
    log_path = os.path.join('__log__', opt.domain)
    
#     if not os.path.exists(res_path):
#         os.makedirs(res_path)
#     if not os.path.exists(log_path):
#         os.makedirs(log_path)

    create_path(res_path)
    create_path(log_path)

    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
#     res_file_name = opt.heuristic + '_trail' + str(opt.trial) + '_' + time_stamp + '.pickle'
#     log_file_name = opt.heuristic + '_trail' + str(opt.trial) + '_' + time_stamp + '.txt'
    
    res_file_name = 'res_'+opt.heuristic + '_trail' + str(opt.trial) + '.pickle'
    pred_file_name = 'pred_'+opt.heuristic + '_trail' + str(opt.trial) + '.h5'
    log_file_name = 'log_'+opt.heuristic + '_trail' + str(opt.trial) + '.txt'
    
    logger = open(os.path.join(log_path, log_file_name), 'w+') 
    
    opt.logger = logger
    
    hlayers_list = []
    for i in range(len(opt.hlayers_w)):
        w = opt.hlayers_w[i]
        d = opt.hlayers_d[i]
        hlayers_list.append([w]*d)  
    opt.hlayers_list = hlayers_list

    synD = dataset.Dataset(opt.domain, opt.trial)
    

#     model_dict = {
#         'mutual':MFDNN, 
#         'dropout': DropoutMFDNN, 
#         'bald': BaldMFDNN, 
#         'full_dropout':FullDropoutMFDNN,
#         'full_bald': FullBaldMFDNN,
#     }
    
    print('initial mf-model with', opt.heuristic)
    
    batch_methods = ['BatchDMFAL', 'DoubleBatchDMFAL', 'BoundBatchDMFAL', 'DoubleBoundBatchDMFAL', 'CycleBoundBatchDMFAL']
    single_methods = ['MutualDMFAL', 'MutualDMFAL-F1', 'MutualDMFAL-F2', 'MutualDMFAL-F3']
    pdv_methods = ['PdvDMFAL', 'PdvDMFAL-F1', 'PdvDMFAL-F2', 'PdvDMFAL-F3']
    
    
        
    if opt.heuristic in batch_methods: 
        model = BatchDMFAL(opt, synD)
    elif opt.heuristic in single_methods:
        model = MutualDMFAL(opt, synD)
    elif opt.heuristic == 'Bald':
        model = BaldAL(opt, synD)
    elif opt.heuristic in pdv_methods:
        model = PdvDMFAL(opt, synD)
    else:
        raise Exception('Error! Invalid heuristic:', opt.heuristic)

    hist_cost = []
    hist_test_nRmse = []
    hist_test_nRmse_ground = []
    hist_N_pred = []
    
    accum_cost = np.sum(np.array(opt.penalty) * np.array(synD.Ntrain_list))
    hist_cost.append(accum_cost)

    for t in trange(opt.T):
        
        t_init = time.time()
        
        
        opt.logger.write('#############################################################\n')
        opt.logger.write('                     Active Step #' + str(t)+'\n')
        opt.logger.write('#############################################################\n')
        opt.logger.flush()

        curr_res = model.train()
        t_train = time.time()-t_init;

        
        if opt.heuristic == 'BatchDMFAL':
            np_X_batch, m_batch = model.batch_query()
            
            t_query = time.time()-t_init-t_train
            
            for j in range(len(m_batch)):
                synD.append(np_X_batch[j], m_batch[j])
            #
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write(' * Time cost per query:' + str((t_query+t_train)/len(np_X_batch)) + '\n')
            opt.logger.write(' * Time cost per append:' + str(t_append/len(np_X_batch)) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
            
        elif opt.heuristic == 'DoubleBatchDMFAL':
            np_X_batch, m_batch = model.batch_double_query()
            
            for j in range(len(m_batch)):
                synD.append(np_X_batch[j], m_batch[j])
            #
        elif opt.heuristic == 'BoundBatchDMFAL':
            np_X_batch, m_batch = model.cycle_bound_batch_query(K=1)
         
            t_query = time.time()-t_init-t_train             
            for j in range(len(m_batch)):
                synD.append(np_X_batch[j], m_batch[j])
            #
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write(' * Time cost per query:' + str((t_query+t_train)/len(np_X_batch)) + '\n')
            opt.logger.write(' * Time cost per append:' + str(t_append/len(np_X_batch)) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        elif opt.heuristic == 'CycleBoundBatchDMFAL':
            np_X_batch, m_batch = model.cycle_bound_batch_query(K=3)

            t_query = time.time()-t_init-t_train           
            for j in range(len(m_batch)):
                synD.append(np_X_batch[j], m_batch[j])
            #
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write(' * Time cost per query:' + str((t_query+t_train)/len(np_X_batch)) + '\n')
            opt.logger.write(' * Time cost per append:' + str(t_append/len(np_X_batch)) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        elif opt.heuristic == 'DoubleBoundBatchDMFAL':
            np_X_batch, m_batch = model.bound_batch_double_query()

            for j in range(len(m_batch)):
                synD.append(np_X_batch[j], m_batch[j])
            #
        elif opt.heuristic == 'MutualDMFAL':
            argx, argm = model.single_query()
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'MutualDMFAL-F1':
            argx, argm = model.single_query_fix(0)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'MutualDMFAL-F2':
            argx, argm = model.single_query_fix(1)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'MutualDMFAL-F3':
            argx, argm = model.single_query_fix(2)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'Bald':
            argx, argm = model.single_query(opt.penalty)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'PdvDMFAL':
            argx, argm = model.single_query()
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'PdvDMFAL-F1':
            argx, argm = model.single_query_fix(0)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'PdvDMFAL-F2':
            argx, argm = model.single_query_fix(1)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        elif opt.heuristic == 'PdvDMFAL-F3':
            argx, argm = model.single_query_fix(2)
            
            t_query = time.time()-t_init-t_train
            synD.append(argx, argm)
            t_append = time.time()-t_init-t_train-t_query
            
            opt.logger.write('------------------------------------------------\n')
            opt.logger.write(' * Train time:' + str(t_train) + '\n')
            opt.logger.write(' * Total query time:' + str(t_query) + '\n')
            opt.logger.write(' * Append time:' + str(t_append) + '\n')
            opt.logger.write('------------------------------------------------\n')
            
        #
        else:
            
            raise Exception('Error: Unrecognized active heuristic!')
            
        accum_cost = np.sum(np.array(opt.penalty) * np.array(synD.Ntrain_list))
        hist_cost.append(accum_cost)
        hist_test_nRmse.append(curr_res['test_rmse'])
        hist_test_nRmse_ground.append(curr_res['test_ground_rmse'])
        
        act_res = {}
        act_res['hist_cost'] = hist_cost
        act_res['hist_test_nRmse'] = hist_test_nRmse
        act_res['hist_test_nRmse_ground'] = hist_test_nRmse_ground
        
        dump_file = open(os.path.join(res_path, res_file_name), "wb")
        pickle.dump(act_res, dump_file)
        dump_file.close()

        
if __name__=='__main__':
    fire.Fire()