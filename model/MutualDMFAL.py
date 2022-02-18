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

class MutualDMFAL(DeepMFNet):
    
    def __init__(self, opt, synD):
        super(MutualDMFAL, self).__init__(opt, synD)
        
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

    
    def eval_joint_base_variance(self, Xq, m):
        # pull out the variables required for approximation
        W_cat_list = []
        S_cat_list = []
        for i in range(self.M):
            W_cat = torch.cat((self.nns_list[i].W_mu, self.nns_list[i].b_mu), dim=0)   # concatenate mean
            S_cat = torch.cat((self.nns_list[i].W_std, self.nns_list[i].b_std), dim=0) # concatenate std
            W_cat_list.append(W_cat)
            S_cat_list.append(S_cat)
        #
        
        # calculate the V[\theta]
        S_flat_list = []
        for S_cat in S_cat_list:
            S_flat = S_cat.reshape([-1])
            #print(S_flat.shape)
            S_flat_list.append(S_flat)
        #
        stack_S_flat = torch.cat(S_flat_list, dim=0)
        
        V_param = torch.diag(torch.square(stack_S_flat))
        # print(V_param.shape)
        
        if self.M==3:
            obj_jac = lambda Wcat0, Wcat1, Wcat2 : self.nonlinear_joint_base(Xq, m, [Wcat0, Wcat1, Wcat2])
        elif self.M==2:
            obj_jac = lambda Wcat0, Wcat1 : self.nonlinear_joint_base(Xq, m, [Wcat0, Wcat1])
        
        jacobians = torch.autograd.functional.jacobian(obj_jac, tuple(W_cat_list), strict=True, create_graph=True)

        # stack the jacobians
        stack_jacobian_list = []
        for Jm in list(jacobians):
            N = Jm.shape[0]
            K = Jm.shape[1]
            mat_flat_Jm = Jm.reshape([N*K, -1])
            stack_jacobian_list.append(mat_flat_Jm)
        #
        J = torch.cat(stack_jacobian_list, dim=1)
        
        V_base = J @ V_param @ J.T # a (K_m+K_M)*N by (K_m+K_M)*N matrix
        
        #print(V_base.shape)
        
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

        
        I_KN = torch.eye(K*N, device=self.device, dtype=self.torch_type)
        
        log_tau = self.log_tau_list[m]
        
        log_det = torch.logdet(torch.exp(log_tau)*torch.matmul(kron_A_Atr_I_N, V_base) + I_KN)
        
        entropy = D*N*(np.log(2*np.pi*np.e)-log_tau) + log_det

        return entropy
    
    def eval_joint_entropy(self, Xq, m):

        V_joint_base = self.eval_joint_base_variance(Xq, m)
    
        # print(V_joint_base.shape)
        
        Am = self.nns_list[m].A
        AM = self.nns_list[self.M-1].A
        
        log_tau_m = self.log_tau_list[m]
        log_tau_M = self.log_tau_list[self.M-1]
        
        Am_hat = torch.exp(log_tau_m)*Am
        AM_hat = torch.exp(log_tau_M)*AM
        
        pad_mM = torch.zeros([Am.shape[0], AM.shape[1]], device=self.device, dtype=self.torch_type)
        pad_Mm = torch.zeros([AM.shape[0], Am.shape[1]], device=self.device, dtype=self.torch_type)
        
        A_u = torch.cat([Am, pad_mM], dim=1)
        A_l = torch.cat([pad_Mm, AM], dim=1)
        A = torch.cat([A_u, A_l], dim=0)
        
        A_hat_u = torch.cat([Am_hat, pad_mM], dim=1)
        A_hat_l = torch.cat([pad_Mm, AM_hat], dim=1)
        A_hat = torch.cat([A_hat_u, A_hat_l], dim=0)
        
        A_hat_Atr = torch.matmul(A_hat, A.T)
        
        N = Xq.shape[0]
        K = self.base_dims[m]
        D = self.output_dims[m]
        
        I_N = torch.eye(N, device=self.device, dtype=self.torch_type)
        
        kron_A_hat_Atr_I_N = utils.Kronecker(A_hat_Atr, I_N)
        
        # print(kron_A_hat_Atr_I_N.shape)
        
        output_var = torch.matmul(kron_A_hat_Atr_I_N, V_joint_base)
        
        I_2KN  = torch.eye(output_var.shape[0], device=self.device, dtype=self.torch_type)
        
#         sign, log_abs_det = torch.slogdet(output_var + I_2KN)
        
#         log_det = torch.log(sign*torch.exp(log_abs_det))
        
        log_det = torch.logdet(output_var + I_2KN)
        
        entropy = log_det + D*N*(2*np.log(2*np.pi*np.e) - (log_tau_m+log_tau_M))
        
        return entropy
    

    
    def eval_query_mutual_info(self, Xquery, m):
        
        if m == self.M-1:
            return self.eval_marginal_entropy(Xquery, m)
        else:
            H_m = self.eval_marginal_entropy(Xquery, m)
            H_M = self.eval_marginal_entropy(Xquery, self.M-1)
            H_mM = self.eval_joint_entropy(Xquery, m)
            
            return H_m + H_M - H_mM

#         H_m = self.eval_marginal_entropy(Xquery, m)
#         H_M = self.eval_marginal_entropy(Xquery, self.M-1)
#         H_mM = self.eval_joint_entropy(Xquery, m)

#         return H_m + H_M - H_mM
    
#         H_m = self.eval_marginal_entropy(Xquery, m)
#         return H_m

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
    

