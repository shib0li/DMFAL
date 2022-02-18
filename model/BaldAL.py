import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
# from tqdm.notebook import tqdm

import time

from config import opt
import model.Utils as utils

# import data.dataset_active as dataset
from model.BaseNet import AdaptiveBaseNet

from infrastructure.misc import *


class BaldAL:
    
    def __init__(self, opt, synD):
        
        self.data = synD
        
        self.logger = opt.logger
        self.verbose = opt.verbose
        
        self.M = opt.M
        self.input_dims = opt.input_dim_list
        self.output_dims = opt.output_dim_list
        self.base_dims = opt.base_dim_list
        self.hlayers = opt.hlayers_list

        self.device = torch.device(opt.placement)
        self.torch_type = opt.torch_type
        
        self.max_epochs = opt.max_epochs
        self.print_freq = opt.print_freq
        self.activation = opt.activation
        self.opt_lr = opt.opt_lr
        
        self.nns_list, self.nns_params_list, self.log_tau_list = self.init_model_params()
        
        self.reg_strength = opt.reg_strength
        self.learning_rate = opt.learning_rate
        
        self.Nquery = opt.Nquery
        

        
        
    
    
    def init_model_params(self,):
        nns_list = []
        nns_params_list = []
        log_tau_list = []
        
        for m in range(self.M):
            if m == 0:
                in_dim = self.input_dims[m]
            else:
                in_dim = self.input_dims[m] + self.base_dims[m-1]
            #
            layers = [in_dim] + self.hlayers[m] + [self.base_dims[m]] + [self.output_dims[m]]
            print(layers)
            nn = AdaptiveBaseNet(layers, self.activation, device=self.device, torch_type=self.torch_type)
            nn_params = nn.parameters()
            log_tau = torch.tensor(0.0, device=self.device, requires_grad=True, dtype=self.torch_type)
            
            nns_list.append(nn)
            nns_params_list.append(nn_params)
            log_tau_list.append(log_tau)
        #
        
        return nns_list, nns_params_list, log_tau_list
    
    def forward(self, X, m, sample=False):
        # first fidelity
        Y_m, base_m = self.nns_list[0].forward(X, sample)
        
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            X_concat = torch.cat((base_m, X), dim=1)
            # print(X_concat.shape)
            Y_m, base_m = self.nns_list[i].forward(X_concat, sample)
        #
        
        return Y_m, base_m

    
    def eval_llh(self, X, Y, m):
        Ns = 5
        llh_samples_list = []
        
        # dist_noise = distributions.normal.Normal(loc=0.0, scale=1/torch.exp(self.log_tau_list[m]))
        
        for ns in range(Ns):
            pred_sample, _ = self.forward(X, m, sample=True)
            
            # log_prob_verify = torch.sum(dist_noise.log_prob(Y-pred_sample))
            # print(log_prob_verify)
            
            log_prob_sample = torch.sum(-0.5*torch.square(torch.exp(self.log_tau_list[m]))*torch.square(pred_sample-Y) +\
                                  self.log_tau_list[m] - 0.5*np.log(2*np.pi))
            
            # print(log_prob_sample)
            
            llh_samples_list.append(log_prob_sample)
        #
        
        return sum(llh_samples_list)

    def batch_eval_llh(self, X_list, Y_list):
        llh_list = []
        for m in range(self.M):
            llh_m = self.eval_llh(X_list[m], Y_list[m], m)
            llh_list.append(llh_m)
        #
        return sum(llh_list)
    
    def batch_eval_kld(self,):
        kld_list = []
        for m in range(self.M):
            kld_list.append(self.nns_list[m]._eval_kld())
        #
        return sum(kld_list)
    
    def batch_eval_reg(self,):
        reg_list = []
        for m in range(self.M):
            reg_list.append(self.nns_list[m]._eval_reg())
        #
        return sum(reg_list)
    
    def eval_rmse_loss(self, X, Y, m):
        pred, _ = self.forward(X, m, sample=False)
        rmse = torch.sqrt(torch.mean(torch.square(Y - pred)))
        return rmse
        
    def batch_eval_rmse(self, X_list, Y_list):
        rmse_list = []
        for m in range(self.M):
            rmse = self.eval_rmse_loss(X_list[m], Y_list[m], m)
            rmse_list.append(rmse)
        #
        return rmse_list
    
    def eval_mae_loss(self, X, Y, m):
        pred, _ = self.forward(X, m, sample=False)
        mae = torch.mean(torch.abs(Y - pred))
        return mae
        
    def batch_eval_mae(self, X_list, Y_list):
        mae_list = []
        for m in range(self.M):
            mae = self.eval_mae_loss(X_list[m], Y_list[m], m)
            mae_list.append(mae)
        #
        return mae_list

    def init_train_optimizer(self, lr, weight_decay):
        opt_params = []
        
        for m in range(self.M):
            
            for nn_param_name, nn_param in self.nns_params_list[m].items():
                # print(nn_param_name)
                opt_params.append({'params':nn_param, 'lr':lr})
            #
            opt_params.append({'params':self.log_tau_list[m], 'lr':lr})
            
        #
        
        return Adam(opt_params, lr=lr, weight_decay=weight_decay)
    
    def eval_rmse(self, m, N_X, N_Y, train=True):
        # inputs are normalized
        N_pred, _ = self.forward(N_X, m, sample=False)
        scales = self.data.get_scales(m, train)
        
        Y = N_Y*scales['y_std'] + scales['y_mean']
        pred = N_pred*scales['y_std'] + scales['y_mean']
        
        rmse = torch.sqrt(torch.mean(torch.square(Y-pred)))
        n_rmse = rmse/scales['y_std']
        
        return rmse.data.cpu().numpy(), n_rmse.data.cpu().numpy()
    
    def eval_rmse_ground(self, m, N_X, np_y_ground, train=True):
        # inputs are normalized
        N_pred, _ = self.forward(N_X, m, sample=False)
        scales = self.data.get_scales(m, train)
        
        mu = np.mean(np_y_ground)
        sig = np.std(np_y_ground)

#         np_N_y_ground = (np_y_ground - np.mean(np_y_ground))/np.std(np_y_ground)

        np_N_pred = N_pred.data.cpu().numpy()
        interp_np_N_pred = self.data.interp_to_ground(np_N_pred, m)
        
        interp_np_pred = interp_np_N_pred*sig + mu
        
        rmse = np.sqrt(np.mean(np.square(np_y_ground-interp_np_pred)))
        n_rmse = rmse/sig
        
        return rmse, n_rmse

    def train(self,):
        
        if self.verbose:
            print('train the model ...')
        
        X_train_list = []
        y_train_list = []
        np_y_train_ground_list = []
        
        X_test_list = []
        y_test_list = []
        np_y_test_ground_list = []
        
        for m in range(self.M):
            
            np_X_train, np_y_train, np_y_train_ground = self.data.get_data(m,train=True, normalize=True, noise=0.01)
            np_X_test, np_y_test, np_y_test_ground = self.data.get_data(m,train=False, normalize=True, noise=0.00)
        
            X_train_list.append(torch.tensor(np_X_train, device=self.device, dtype=self.torch_type))
            y_train_list.append(torch.tensor(np_y_train, device=self.device, dtype=self.torch_type))
            np_y_train_ground_list.append(np_y_train_ground)
            
            X_test_list.append(torch.tensor(np_X_test, device=self.device, dtype=self.torch_type))
            y_test_list.append(torch.tensor(np_y_test, device=self.device, dtype=self.torch_type))
            np_y_test_ground_list.append(np_y_test_ground)
            
        #
        
        hist_test_rmse = []
        hist_test_ground_rmse = []
        
        optimizer_train = self.init_train_optimizer(self.learning_rate, 0.0)
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs+1):

            optimizer_train.zero_grad()
            loss = -self.batch_eval_llh(X_train_list, y_train_list) + self.batch_eval_kld() + self.reg_strength*self.batch_eval_reg()
            loss.backward(retain_graph=True)
            optimizer_train.step()
            
            if epoch % self.print_freq == 0:
                
                if self.verbose:
                    print('======================================')
                    print('%d-th epoch: loss=%.7f' % (epoch, loss))
                    print('======================================')
                self.logger.write('=============================================================\n')
                self.logger.write(str(epoch) + '-th epoch: loss=' + str(loss.data.cpu().numpy()) +\
                                  ', time_elapsed:' + str(time.time()-start_time) + '\n')
                self.logger.write('=============================================================\n')
                
                buff_test_nRmse = []
                buff_test_nRmse_ground = []

                for m in range(self.M):

                    train_rmse, n_train_rmse = self.eval_rmse(m, X_train_list[m], y_train_list[m], train=True)
                    test_rmse, n_test_rmse = self.eval_rmse(m, X_test_list[m], y_test_list[m], train=False)
                    
                    train_ground_rmse, n_train_ground_rmse = self.eval_rmse_ground(
                        m, X_train_list[m], np_y_train_ground_list[m], train=True)
                    test_ground_rmse, n_test_ground_rmse = self.eval_rmse_ground(
                        m, X_test_list[m], np_y_test_ground_list[m], train=False)
                    
                    buff_test_nRmse.append(n_test_rmse)
                    buff_test_nRmse_ground.append(n_test_ground_rmse)
       
                    if self.verbose:
                        print('  m=%d:' % (m))
                        print('  * (origin) train_rmse=%.7f, test_rmse=%.7f' % (n_train_rmse, n_test_rmse))
                        print('  * (ground) train_rmse=%.7f, test_rmse=%.7f' % (n_train_ground_rmse, n_test_ground_rmse))
#                         print('  * (ground) train_rmse=%.7f, test_rmse=%.7f' % (train_ground_rmse, test_ground_rmse))
                    # if verbose
                    self.logger.write('m='+str(m)+'\n')
                    self.logger.write('  * (origin) train_rmse='+str(n_train_rmse)+', test_rmse='+str(n_test_rmse)+'\n')
                    self.logger.write('  * (ground) train_rmse='+str(n_train_ground_rmse)+',test_rmse='+str(n_test_ground_rmse)+'\n')
                    self.logger.write('  * log_tau_m='+str(self.log_tau_list[m].data.cpu().numpy())+'\n')
                # for m
                
                hist_test_rmse.append(np.array(buff_test_nRmse))
                hist_test_ground_rmse.append(np.array(buff_test_nRmse_ground))
                
            # if epoch
            self.logger.flush()
        # for epoch
        
        N_pred, _ = self.forward(X_test_list[-1], self.M-1, sample=False)
        
        res = {}
        res['test_rmse'] = np.array(hist_test_rmse)
        res['test_ground_rmse'] = np.array(hist_test_ground_rmse)
        res['N_predict'] = N_pred.data.cpu().numpy()

        return res

    def nonlinear_marginal_base(self, X, m, Wcat_list):
        # first fidelity
        W = Wcat_list[0][0:-1, :]
        b = Wcat_list[0][-1, :].reshape([1,-1])
        #print(W.shape)
        #print(b.shape)
        base_m = self.nns_list[0].forward_base_by_sample(X, W, b)
        
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            W = Wcat_list[i][0:-1, :]
            b = Wcat_list[i][-1, :].reshape([1,-1])
            #print(W.shape)
            #print(b.shape)

            X_concat = torch.cat((base_m, X), dim=1)
            base_m = self.nns_list[i].forward_base_by_sample(X_concat, W, b)
        #
        return base_m

    def eval_marginal_base_variance(self, Xq, m):
        # pull out the variables required for approximation
        pred_samples_list = []
        Ns = 10
        for ns in range(Ns):
            pred, base = self.forward(Xq, m, sample=True)
            pred_samples_list.append(base)
        #
        
        pred_samples = torch.stack(pred_samples_list)
        pred_samples = torch.reshape(pred_samples, [Ns, -1, 1])
        
        mu = torch.mean(pred_samples, dim=0)
        
        diff = pred_samples - mu
        
        diff_tr = diff.permute(0,2,1)
        
        V_base = torch.sum(torch.einsum('bij,bjk->bik', diff, diff_tr), dim=0)/(Ns-1)
        
        return V_base

    def eval_marginal_entropy(self, Xq, m):
        
        V_base = self.eval_marginal_base_variance(Xq, m)
        A = self.nns_list[m].A
        
        N = Xq.shape[0]
        
        K = self.base_dims[m]
        D = self.output_dims[m]
        
        I_N = torch.eye(N, device=self.device, dtype=self.torch_type)
        
        A_Atr = torch.matmul(A, A.T)
        
        kron_A_Atr_I_N = utils.Kronecker(A_Atr, I_N)
        
        # print(V_base.shape)
        # print(kron_A_Atr_I_N.shape)
        
        I_KN = torch.eye(K*N, device=self.device, dtype=self.torch_type)
        
        log_tau = self.log_tau_list[m]
        
#         sign, log_abs_det = torch.slogdet(torch.exp(log_tau)*torch.matmul(kron_A_Atr_I_N, V_base) + I_KN)
#         log_det = torch.log(sign*torch.exp(log_abs_det))

        log_det = torch.logdet(torch.exp(log_tau)*torch.matmul(kron_A_Atr_I_N, V_base) + I_KN)
        
        entropy = D*N*(np.log(2*np.pi*np.e)-log_tau) + log_det

        return entropy
    
    
    def eval_query_mutual_info(self, Xquery, m):
    
        H_m = self.eval_marginal_entropy(Xquery, m)
        return H_m

    def init_query_points(self, Nq, m):
        lb, ub = self.data.get_N_bounds(m)
        scale = (ub-lb).reshape([1,-1])
        uni_noise = np.random.uniform(size=[Nq, self.input_dims[m]])
        
        np_Xq_init = uni_noise*scale + lb
        
        Xq = torch.tensor(np_Xq_init, device=self.device, dtype=self.torch_type, requires_grad=True)
        
        return Xq


    def eval_query(self, m):
        
        # sometimes the log det will throw numerical erros, if that happens, re-try

        mutual_info = None
        Xq = None
        
        max_retry = 10
        count = 0
        success = False

        while not success:

            if count <= max_retry:
                try:

                    Xq = self.init_query_points(self.Nquery, m)

                    np_lb, np_ub = self.data.get_N_bounds(m)
                    bounds = torch.tensor(np.vstack((np_lb, np_ub)), device=self.device, dtype=self.torch_type)

                    optimizer_query = LBFGS([Xq], self.opt_lr)

                    mutual_info = self.eval_query_mutual_info(Xq, m)
                    if self.verbose:
                        print('Query m=%d info BEFORE Opt'%(m))
                        print('  - info:  ', mutual_info.data.cpu().numpy())
                        print('  - query: ', Xq.data.cpu().numpy())
                        
                    self.logger.write("start to query fidelity m=" + str(m) + '\n')
                    self.logger.write("  - info BEFORE " + str(mutual_info.data.cpu().numpy()) + '\n')
                    self.logger.write("  - Xq   BEFORE " + str(Xq.data.cpu().numpy()) + '\n')

                    def closure():
                        optimizer_query.zero_grad()  
                        loss = -self.eval_query_mutual_info(Xq, m)
                        loss.backward(retain_graph=True)

                        with torch.no_grad():
                            for j, (lb, ub) in enumerate(zip(*bounds)):
                                Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                            #
                        #
                        return loss

                    optimizer_query.step(closure)

                    mutual_info = self.eval_query_mutual_info(Xq, m)
                    if self.verbose:
                        print('Query m=%d info AFTER Opt'%(m))
                        print('  - info:  ', mutual_info.data.cpu().numpy())
                        print('  - query: ', Xq.data.cpu().numpy())
                        
                    self.logger.write("  - info AFTER " + str(mutual_info.data.cpu().numpy()) + '\n')
                    self.logger.write("  - Xq   AFTER " + str(Xq.data.cpu().numpy()) + '\n')
                        

                    if mutual_info < 0:
                        if self.verbose:
                            print('MI < 0, give another try... count', count)
                        self.logger.write('MI < 0, give another try... count ' + str(count) + '\n')
                        self.logger.flush()
                        success=False
                        count += 1
                    else:
                        success=True
                    #
                except:
                    if self.verbose:
                        print('Opt fails, give another try... count', count)
                    self.logger.write('Opt fails, give another try... count ' + str(count)+'\n')
                    self.logger.flush()
                    success=False
                    count += 1
            
                # try
            else:
                success=True
                Xq = self.init_query_points(self.Nquery, m)
                mutual_info = torch.tensor(0.0)
            #if
        # while

        return mutual_info, Xq
    

#     def batch_query(self, penalties):
#         mutual_info_list = []
#         query_list = []
#         for m in range(self.M):
#             mutul_info, Xq = self.eval_query(m)
#             mutual_info_list.append(mutul_info.data.cpu().numpy())
#             query_list.append(Xq.data.cpu().numpy())
#         #
#         reg_mutual_info_list = np.array(mutual_info_list)/np.array(penalties)
        
#         argm = np.argmax(reg_mutual_info_list)
#         argx = query_list[argm]
        
#         return mutual_info_list, query_list, argm, argx

    def single_query(self, penalties):
        mutual_info_list = []
        query_list = []
        for m in range(self.M):
            mutul_info, Xq = self.eval_query(m)
            mutual_info_list.append(mutul_info.data.cpu().numpy())
            query_list.append(Xq.data.cpu().numpy())
        #
        reg_mutual_info_list = np.array(mutual_info_list)/np.array(penalties)
        
        argm = np.argmax(reg_mutual_info_list)
        argx = query_list[argm]
        
        self.logger.write('argm='+str(argm)+'\n')
        self.logger.write('argx='+str(argx)+'\n')
        self.logger.flush()
        
        return argx, argm

#     def debug(self,):
#         print('debug mode ...')
        
#         penalties = [1,1,1]
        
#         mutual_info_list, query_list, argm, argx = self.batch_query(penalties)
        
#         print(mutual_info_list)
#         print(query_list)
#         print(argm)
#         print(argx)



        

        
        
        
    
    
        
# Ntrain_list = [10,5,2]
# Ntest_list = [10,5,2]
# Domain = 'Heat_1D'

# synD = dataset.Dataset(Ntrain_list, Ntest_list, Domain)

# model = MFDNN(opt, synD)

# model.train()

# model.debug()