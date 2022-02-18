import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import LBFGS
import time

from model.DeepMFNet import DeepMFNet

class BatchDMFAL(DeepMFNet):
    
    def __init__(self, opt, synD):
        super().__init__(opt, synD)
        self.Nquery = 1

        self.batch_size = opt.batch_size
        self.costs = np.array(opt.penalty)


    def init_query_points(self, Nq, m):
        lb, ub = self.data.get_N_bounds(m)
        scale = (ub-lb).reshape([1,-1])
        uni_noise = np.random.uniform(size=[Nq, self.input_dims[m]])
        
        np_Xq_init = uni_noise*scale + lb
        
        Xq = torch.tensor(np_Xq_init, device=self.device, dtype=self.torch_type, requires_grad=True)
        
        return Xq

    def single_nonlinear_base(self, X, m, weights_list):
        # first fidelity
        W = weights_list[0][0:-1, :]
        b = weights_list[0][-1, :].reshape([1,-1])
        #print(W.shape)
        #print(b.shape)
        base_m = self.nns_list[0].forward_base_by_sample(X, W, b)
        
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            W = weights_list[i][0:-1, :]
            b = weights_list[i][-1, :].reshape([1,-1])
            #print(W.shape)
            #print(b.shape)

            X_concat = torch.cat((base_m, X), dim=1)
            base_m = self.nns_list[i].forward_base_by_sample(X_concat, W, b)
        #
        return base_m

    def batch_nonlinear_base(self, X_batch, m_batch, weights_list):
        n = len(X_batch)
        batch_base_list = []
        for j in range(n):
            base_j = self.single_nonlinear_base(X_batch[j], m_batch[j], weights_list)
            #print(base_j.shape)
            batch_base_list.append(base_j)
        #    
        batch_base = torch.cat(batch_base_list, dim=1)
        #print(batch_base.shape)
        return batch_base

    def eval_batch_base_variance(self, X_batch, m_batch):
        weights_list = []
        var_list = []
        
        hf = np.max(np.array(m_batch)) + 1
        
        for i in range(hf):
            Wcat = torch.cat((self.nns_list[i].W_mu, self.nns_list[i].b_mu), dim=0)   # concatenate mean
            #print(Wcat.shape)
            Scat = torch.cat((self.nns_list[i].W_std, self.nns_list[i].b_std), dim=0) # concatenate std
            #print(Scat.shape)
            weights_list.append(Wcat)
            var_list.append(Scat)
        #
        
        # flatten the variance
        flat_var_list = []
        for var in var_list:
            flat_var = var.reshape([-1])
            #print(flat_var.shape)
            flat_var_list.append(flat_var)
        #
        stack_flat_var = torch.cat(flat_var_list, dim=0)
        
        V_param = torch.diag(torch.square(stack_flat_var))
        #print(V_param.shape)

        # calculate the jacobians
        # objects used to run batch jacobians
        if hf == 1:
            obj_func = lambda Wcat0 : self.batch_nonlinear_base(X_batch, m_batch, [Wcat0])
        elif hf == 2:
            obj_func = lambda Wcat0, Wcat1 : self.batch_nonlinear_base(X_batch, m_batch, [Wcat0, Wcat1])
        elif hf == 3:
            obj_func = lambda Wcat0, Wcat1, Wcat2 : self.batch_nonlinear_base(X_batch, m_batch, [Wcat0, Wcat1, Wcat2])
        #
    
    
        jacobians = torch.autograd.functional.jacobian(obj_func, tuple(weights_list), strict=True, create_graph=True)
        
        # stack the jacobians
        stack_jacobian_list = []
        for Jm in list(jacobians):
            N = Jm.shape[0]
            K = Jm.shape[1]
            mat_flat_Jm = Jm.reshape([N*K, -1])
            #print(mat_flat_Jm.shape)
            stack_jacobian_list.append(mat_flat_Jm)
        #
        J = torch.cat(stack_jacobian_list, dim=1)
        #print(J.shape)
        
        V_base = J @ V_param @ J.T # a KN by KN matrix
        #print(V_base.shape)
        
        return V_base
    
    def eval_batch_output_entropy(self, X_batch, m_batch):
        V_batch_base = self.eval_batch_base_variance(X_batch, m_batch)
        Am_list = []
        Am_hat_list = []
        log_tau_m_list = []
        K_list = []
        
        D_list = []
        D_log_tau_list = []
        
        for m in m_batch:
            Am_list.append(self.nns_list[m].A)
            log_tau_m_list.append(self.log_tau_list[m])
            Am_hat_list.append(torch.exp(self.log_tau_list[m])*self.nns_list[m].A)
            K_list.append(self.base_dims[m])
            
            D_list.append(self.output_dims[m])
            D_log_tau_list.append(self.output_dims[m]*self.log_tau_list[m])
        #
        
        A = torch.block_diag(*Am_list)
        A_hat = torch.block_diag(*Am_hat_list)
        
        A_hat_A_tr = torch.matmul(A_hat, A.T)
        
        I_KN = torch.eye(sum(K_list), device=self.device, dtype=self.torch_type)
        
        output_var = torch.matmul(A_hat_A_tr, V_batch_base) + I_KN
        
        log_det = torch.logdet(output_var)
        
        entropy = sum(D_list)*np.log(2*np.pi*np.e) - 0.5*sum(D_log_tau_list) + 0.5*log_det
        
        return entropy
    
    def eval_batch_mutual_info(self, X_batch, m_batch):
        H_batch_m = self.eval_batch_output_entropy(X_batch, m_batch)
        
        M_batch = [self.M-1]*len(m_batch)
        H_batch_M = self.eval_batch_output_entropy(X_batch, M_batch)
        
        H_batch_mM = self.eval_batch_output_entropy(X_batch+X_batch, m_batch+M_batch)
        
        return H_batch_m + H_batch_M - H_batch_mM
    
    def eval_next_query(self, prev_X_batch, prev_m_batch, m):
        
        max_retry = 10
        count = 0
        success = False
        
        while not success:
            if count <= max_retry:
                try:
                    Xq = self.init_query_points(self.Nquery, m)

                    np_lb, np_ub = self.data.get_N_bounds(m)
                    bounds = torch.tensor(np.vstack((np_lb, np_ub)), device=self.device, dtype=self.torch_type)

                    lbfgs = LBFGS([Xq], self.opt_lr)
                    mutual_info = self.eval_batch_mutual_info(prev_X_batch+[Xq], prev_m_batch+[m])

                    if self.verbose:
                        print('Query m=%d info BEFORE Opt'%(m))
                        print('  - info:  ', mutual_info.data.cpu().numpy())
                        print('  - query: ', Xq.data.cpu().numpy())

                    self.logger.write("start to query fidelity m=" + str(m) + '\n')
                    self.logger.write("  - info BEFORE " + str(mutual_info.data.cpu().numpy()) + '\n')
                    self.logger.write("  - Xq   BEFORE " + str(Xq.data.cpu().numpy()) + '\n')


                    def closure():
                        lbfgs.zero_grad()  
                        loss = -self.eval_batch_mutual_info(prev_X_batch+[Xq], prev_m_batch+[m])
                        #print(loss)
                        loss.backward(retain_graph=True)

                        with torch.no_grad():
                            for j, (lb, ub) in enumerate(zip(*bounds)):
                                Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                            #
                        #
                        return loss

                    lbfgs.step(closure)

                    mutual_info = self.eval_batch_mutual_info(prev_X_batch+[Xq], prev_m_batch+[m])
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
                    # if
                except:
                    if self.verbose:
                        print('Opt fails, give another try... count', count)
                    self.logger.write('Opt fails, give another try... count ' + str(count)+'\n')
                    self.logger.flush()
                    success=False
                    count += 1
                # catch
            else:
                success=True
                Xq = self.init_query_points(self.Nquery, m)
                mutual_info = torch.tensor(0.0)
            #if
        # while
        
        return mutual_info, Xq
        
        
    def single_query(self, X_batch_prev, m_batch_prev, acc_cost_prev):

        fidelity_info = []
        fidelity_query = []
        
        for m in range(self.M):
            info, xq = self.eval_next_query(X_batch_prev, m_batch_prev, m)
            fidelity_info.append(info.data.cpu().numpy())
            fidelity_query.append(xq)
        #

        reg_info = np.array(fidelity_info) / (self.costs + acc_cost_prev)
#         print('regularized info', reg_info)
        
        argm = np.argmax(reg_info)
        argx = fidelity_query[argm]
        
        if self.verbose:
            print(argm)
            print(argx)
        self.logger.write('argm='+str(argm)+'\n')
        self.logger.write('argx='+str(argx.data.cpu().numpy())+'\n')
        self.logger.flush()
        
        X_batch_curr = X_batch_prev + [argx]
        m_batch_curr = m_batch_prev + [argm]
        acc_cost_curr = acc_cost_prev + self.costs[argm]
        
        return X_batch_curr, m_batch_curr, acc_cost_curr

#         return argm, argx
    
    def batch_query(self,):

        X_batch = []
        m_batch = []
        acc_costs = 0.0
        
        for j in range(self.batch_size):
            if self.verbose:
                print('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***')
            self.logger.write('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***\n')
            self.logger.flush()
            X_batch, m_batch, acc_costs = self.single_query(X_batch, m_batch, acc_costs)

        np_X_batch = []
        for xq in X_batch:
            np_X_batch.append(xq.data.cpu().numpy())
            
        if self.verbose:
            print(np_X_batch)
            print(m_batch)
            print(acc_costs)
            
        return np_X_batch, m_batch
    
    