import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
import time

import model.ExpUtils as utils
from model.DeepMFNet import DeepMFNet

# import matplotlib.pyplot as plt

# TORCH_TYPE = opt.torch_type
# DEVICE = opt.device
# DEVICE = torch.device(opt.placement)

from infrastructure.misc import *

class PdvDMFAL(DeepMFNet):
    
    def __init__(self, opt, synD):
        super(PdvDMFAL, self).__init__(opt, synD)
        
        self.Nquery = 1
        self.costs = opt.penalty

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

    def nonlinear_joint_base(self, X, m, Wcat_list):
        base_m = self.nonlinear_marginal_base(X, m, Wcat_list[0:m+1])
        base_M = self.nonlinear_marginal_base(X, self.M-1, Wcat_list[0:self.M])
        
        joint_base = torch.cat((base_m, base_M), dim=1)
        
        return joint_base
    
    def eval_marginal_base_variance(self, Xq, m):
        # pull out the variables required for approximation
        W_cat_list = []
        S_cat_list = []
        for i in range(m+1):
            W_cat = torch.cat((self.nns_list[i].W_mu, self.nns_list[i].b_mu), dim=0)   # concatenate mean
            S_cat = torch.cat((self.nns_list[i].W_std, self.nns_list[i].b_std), dim=0) # concatenate std
            W_cat_list.append(W_cat)
            S_cat_list.append(S_cat)
        #

        # calculate the V[\theta]
        S_flat_list = []
        for S_cat in S_cat_list:
            S_flat = S_cat.reshape([-1])
            # print(S_flat.shape)
            S_flat_list.append(S_flat)
        #
        stack_S_flat = torch.cat(S_flat_list, dim=0)
        
        V_param = torch.diag(torch.square(stack_S_flat))
        #print(V_param.shape)
        
        
        # calculate the jacobians
        # objects used to run batch jacobians
        if m == 0:
            obj_jac = lambda Wcat0 : self.nonlinear_marginal_base(Xq, m, [Wcat0])
        elif m == 1:
            obj_jac = lambda Wcat0, Wcat1 : self.nonlinear_marginal_base(Xq, m, [Wcat0, Wcat1])
        elif m == 2:
            obj_jac = lambda Wcat0, Wcat1, Wcat2 : self.nonlinear_marginal_base(Xq, m, [Wcat0, Wcat1, Wcat2])
        #

        jacobians = torch.autograd.functional.jacobian(obj_jac, tuple(W_cat_list), strict=True, create_graph=True)
        
        # stack the jacobians
        stack_jacobian_list = []
        for Jm in list(jacobians):
            N = Jm.shape[0]
            K = Jm.shape[1]
            mat_flat_Jm = Jm.reshape([N*K, -1])
            # print(mat_flat_Jm.shape)
            stack_jacobian_list.append(mat_flat_Jm)
        #
        J = torch.cat(stack_jacobian_list, dim=1)
        # print(J.shape)
        
        V_base = J @ V_param @ J.T # a KN by KN matrix
        
        return V_base


    def eval_predictive_variance(self, Xq, m):
        
        V_base = self.eval_marginal_base_variance(Xq, m)
        A = self.nns_list[m].A
        
        
        
        N = Xq.shape[0]
        
        K = self.base_dims[m]
        D = self.output_dims[m]
        
        I_N = torch.eye(N, device=self.device, dtype=self.torch_type)
        kron_A = utils.Kronecker(A, I_N)
        kron_A_tr = utils.Kronecker(A.T, I_N)
        
        #output_var = kron_A_tr @ V_base @ kron_A
        
        output_var = (A.T) @ V_base @ A
        
        pred_var = torch.mean(torch.diag(output_var))

        return pred_var

    
    def eval_query_info(self, Xquery, m):

        return self.eval_predictive_variance(Xquery, m)


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

                    mutual_info = self.eval_query_info(Xq, m)
                    if self.verbose:
                        print('Query m=%d info BEFORE Opt'%(m))
                        print('  - info:  ', mutual_info.data.cpu().numpy())
                        print('  - query: ', Xq.data.cpu().numpy())
                        
                    self.logger.write("start to query fidelity m=" + str(m) + '\n')
                    self.logger.write("  - info BEFORE " + str(mutual_info.data.cpu().numpy()) + '\n')
                    self.logger.write("  - Xq   BEFORE " + str(Xq.data.cpu().numpy()) + '\n')

                    def closure():
                        optimizer_query.zero_grad()  
                        loss = -self.eval_query_info(Xq, m)
                        loss.backward(retain_graph=True)

                        with torch.no_grad():
                            for j, (lb, ub) in enumerate(zip(*bounds)):
                                Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                            #
                        #
                        return loss

                    optimizer_query.step(closure)

                    mutual_info = self.eval_query_info(Xq, m)
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
    
    def single_query(self,):
        penalties = self.costs
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
    
    def single_query_fix(self, m):

        mutul_info, Xq = self.eval_query(m)
        
        argm = m
        argx = Xq.data.cpu().numpy()
        
        self.logger.write('argm='+str(argm)+'\n')
        self.logger.write('argx='+str(argx)+'\n')
        self.logger.flush()
        
        return argx, argm
    

