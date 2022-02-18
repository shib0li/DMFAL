import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
from torch.optim import LBFGS
import time

from model.DeepMFNet import DeepMFNet

class BatchDMFAL(DeepMFNet):
    
    def __init__(self, opt, synD):
        super().__init__(opt, synD)

        self.batch_size = opt.batch_size
        self.costs = np.array(opt.penalty)
        
        self.concat_weights_mean = []
        self.concat_weights_std = []
        
        for i in range(self.M):
            concat_mu = torch.cat((self.nns_list[i].W_mu, self.nns_list[i].b_mu), dim=0)   # concatenate mean
            concat_std = torch.cat((self.nns_list[i].W_std, self.nns_list[i].b_std), dim=0) # concatenate std
            self.concat_weights_mean.append(concat_mu)
            self.concat_weights_std.append(concat_std)
        #
        
        self.V_param_list = []
        self.param_dims = []
        for m in range(self.M):
            V_param = self.eval_params_var(m)
            self.V_param_list.append(V_param)
            self.param_dims.append(V_param.shape[0])


    def init_query_points(self, m, Nq=1):
        lb, ub = self.data.get_N_bounds(m)
        scale = (ub-lb).reshape([1,-1])
        uni_noise = np.random.uniform(size=[Nq, self.input_dims[m]])
        
        np_Xq_init = uni_noise*scale + lb
        
        Xq = torch.tensor(np_Xq_init, device=self.device, dtype=self.torch_type, requires_grad=True)
        
        return Xq

    def eval_params_var(self, m):
        # flatten the variance
        std_list = self.concat_weights_std[:m+1]
        flat_std_list = []
        for std in std_list:
            flat_std = std.reshape([-1])
            #print(flat_var.shape)
            flat_std_list.append(flat_std)
        #
        stack_flat_std = torch.cat(flat_std_list, dim=0)
        
        V_param = torch.diag(torch.square(stack_flat_std))
        
        return V_param
    
    def single_nonlinear_base(self, X, m, weights_list):
        # first fidelity
        W = weights_list[0][0:-1, :]
        b = weights_list[0][-1, :].reshape([1,-1])

        base_m = self.nns_list[0].forward_base_by_sample(X, W, b)
        
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            W = weights_list[i][0:-1, :]
            b = weights_list[i][-1, :].reshape([1,-1])

            X_concat = torch.cat((base_m, X), dim=1)
            base_m = self.nns_list[i].forward_base_by_sample(X_concat, W, b)
        #
        return base_m
    
    def eval_output_jacob(self, X, m):
        
        weights_list = self.concat_weights_mean[:m+1]
        
        if m == 0:
            obj_func = lambda Wcat0 : self.single_nonlinear_base(X, m, [Wcat0])
        elif m == 1:
            obj_func = lambda Wcat0, Wcat1 : self.single_nonlinear_base(X, m, [Wcat0, Wcat1])
        elif m == 2:
            obj_func = lambda Wcat0, Wcat1, Wcat2 : self.single_nonlinear_base(X, m, [Wcat0, Wcat1, Wcat2])
        #
        
        jacobians = torch.autograd.functional.jacobian(obj_func, tuple(weights_list), strict=True, create_graph=True)
        
        # stack the jacobians
        stack_jacobian_list = []
        for Jm in list(jacobians):
            N = Jm.shape[0]
            K = Jm.shape[1]
            mat_flat_Jm = Jm.reshape([N*K, -1])
            stack_jacobian_list.append(mat_flat_Jm)
        #
        J = torch.cat(stack_jacobian_list, dim=1)
        
        return J

    def eval_batch_base_variance_jacobians(self, X_batch, m_batch, J_batch):

        hf = np.max(np.array(m_batch))
#         print(m_batch)
#         print('hf', hf)
        
        
        V_param = self.V_param_list[hf]
        target_dim = self.param_dims[hf]
        
        pad_J_batch = []
        
        for J in J_batch:
            dim = J.shape[1]
            if dim < target_dim:
                padding = nn.ZeroPad2d((0, target_dim-dim, 0, 0))
                pad_J = padding(J)
                pad_J_batch.append(pad_J)
            else:
                pad_J_batch.append(J)
            #
        
        #
        
        stack_J = torch.cat(pad_J_batch, dim=0)
        V_base = stack_J @ V_param @ stack_J.T
        
        return V_base


    def eval_batch_output_entropy(self, X_batch, m_batch, J_batch):
        
        V_batch_base = self.eval_batch_base_variance_jacobians(X_batch, m_batch, J_batch)
        
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
    
    def eval_batch_mutual_info(self, X_batch, m_batch, J_m_batch, J_M_batch, J_joint_batch):
        
        H_batch_m = self.eval_batch_output_entropy(X_batch, m_batch, J_m_batch)
        
        #print('info_lhs', H_batch_m)
        
        M_batch = [self.M-1]*len(m_batch)
        
        #print(M_batch)
        
        H_batch_M = self.eval_batch_output_entropy(X_batch, M_batch, J_M_batch)
        
        #print('info_rhs', H_batch_M)
        
        #print(m_batch)
        #print(M_batch)
        #print(m_batch+M_batch)
        
        H_batch_mM = self.eval_batch_output_entropy(X_batch+X_batch, m_batch+M_batch, J_joint_batch)
        
        return H_batch_m + H_batch_M - H_batch_mM

    def eval_next_query(self, prev_X_batch, prev_m_batch, prev_J, prev_J_hf, prev_J_joint, m):
        
        max_retry = 10
        count = 0
        success = False

        while not success:
            if count <= max_retry:
                try:
                    Xq = self.init_query_points(m)

                    np_lb, np_ub = self.data.get_N_bounds(m)
                    bounds = torch.tensor(np.vstack((np_lb, np_ub)), device=self.device, dtype=self.torch_type)

                    lbfgs = LBFGS([Xq], self.opt_lr)


                    Jm = self.eval_output_jacob(Xq, m)
                    JM = self.eval_output_jacob(Xq, self.M-1)

                    curr_J = prev_J + [Jm]
                    curr_J_hf = prev_J_hf + [JM]
                    curr_J_joint = curr_J+curr_J_hf
                    mutual_info = self.eval_batch_mutual_info(prev_X_batch+[Xq], prev_m_batch+[m],curr_J,curr_J_hf,curr_J_joint)

                    if self.verbose:
                        print('Query m=%d info BEFORE Opt'%(m))
                        print('  - info:  ', mutual_info.data.cpu().numpy())
                        print('  - query: ', Xq.data.cpu().numpy())

                    self.logger.write("start to query fidelity m=" + str(m) + '\n')
                    self.logger.write("  - info BEFORE " + str(mutual_info.data.cpu().numpy()) + '\n')
                    self.logger.write("  - Xq   BEFORE " + str(Xq.data.cpu().numpy()) + '\n')


                    def closure():
                        lbfgs.zero_grad()  

                        Jm = self.eval_output_jacob(Xq, m)
                        JM = self.eval_output_jacob(Xq, self.M-1)

                        curr_J = prev_J + [Jm]
                        curr_J_hf = prev_J_hf + [JM]
                        curr_J_joint = curr_J+curr_J_hf

                        loss = -self.eval_batch_mutual_info(prev_X_batch+[Xq], prev_m_batch+[m],curr_J,curr_J_hf,curr_J_joint)
                        #print(loss)
                        loss.backward(retain_graph=True)

                        with torch.no_grad():
                            for j, (lb, ub) in enumerate(zip(*bounds)):
                                Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                            #
                        #
                        return loss

                    lbfgs.step(closure)


                    Jm = self.eval_output_jacob(Xq, m)
                    JM = self.eval_output_jacob(Xq, self.M-1)

                    curr_J = prev_J + [Jm]
                    curr_J_hf = prev_J_hf + [JM]
                    curr_J_joint = curr_J+curr_J_hf

                    mutual_info = self.eval_batch_mutual_info(prev_X_batch+[Xq], prev_m_batch+[m],curr_J,curr_J_hf,curr_J_joint)
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
                Xq = self.init_query_points(m)
                mutual_info = torch.tensor(0.0)
            #if
        # while
        
        return mutual_info, Xq
    
    def single_query(self, prev_X_batch, prev_m_batch, prev_acc_cost, prev_J_m_batch, prev_J_M_batch, prev_J_joint_batch):

        fidelity_info = []
        fidelity_query = []
        
        for m in range(self.M):
            info, xq = self.eval_next_query(
                prev_X_batch, prev_m_batch, prev_J_m_batch, prev_J_M_batch, prev_J_joint_batch, m)
            
            fidelity_info.append(info.data.cpu().numpy())
            fidelity_query.append(xq)
        #

        reg_info = np.array(fidelity_info) / (self.costs + prev_acc_cost)

        
        argm = np.argmax(reg_info)
        argx = fidelity_query[argm]
        
        if self.verbose:
            print(argm)
            print(argx)
        self.logger.write('argm='+str(argm)+'\n')
        self.logger.write('argx='+str(argx.data.cpu().numpy())+'\n')
        self.logger.flush()
        
        Jm = self.eval_output_jacob(argx, argm)
        JM = self.eval_output_jacob(argx, self.M-1)
            
        curr_J_m_batch = prev_J_m_batch + [Jm]
        curr_J_M_batch = prev_J_M_batch + [JM]
        curr_J_joint_batch = curr_J_m_batch+curr_J_M_batch
        
        curr_X_batch = prev_X_batch + [argx]
        curr_m_batch = prev_m_batch + [argm]
        curr_acc_cost = prev_acc_cost + self.costs[argm]
        
        return curr_X_batch, curr_m_batch, curr_acc_cost, curr_J_m_batch, curr_J_M_batch, curr_J_joint_batch 

    def single_query_by_fidelity(self, prev_X_batch, prev_m_batch, prev_acc_cost, 
                                 prev_J_m_batch, prev_J_M_batch, prev_J_joint_batch, fidelity):

        fidelity_info = []
        fidelity_query = []
        
        m = fidelity 
        info, xq = self.eval_next_query(
            prev_X_batch, prev_m_batch, prev_J_m_batch, prev_J_M_batch, prev_J_joint_batch, m)
        
        if self.verbose:
            print(m)
            print(xq)
        self.logger.write('argm='+str(m)+'\n')
        self.logger.write('argx='+str(xq.data.cpu().numpy())+'\n')
        self.logger.flush()
        
        Jm = self.eval_output_jacob(xq, m)
        JM = self.eval_output_jacob(xq, self.M-1)
        
        return xq, info, Jm, JM
    
    def double_query(self, prev_X_batch, prev_m_batch, prev_acc_cost, 
                                 prev_J_m_batch, prev_J_M_batch, prev_J_joint_batch):

        reg_info_gain_mat = np.zeros(shape=[self.M, self.M])
        Xq_hist = []
        Jm_hist = []
        JM_hist = []
        
        for m_i in range(self.M):
            
            Xq_i, info_i, Jm_i, JM_i = self.single_query_by_fidelity(
                prev_X_batch, prev_m_batch, prev_acc_cost, prev_J_m_batch, prev_J_M_batch, prev_J_joint_batch, m_i)
            
            buff_Jm = prev_J_m_batch.copy()
            buff_JM = prev_J_M_batch.copy()
            
            buff_Jm = buff_Jm + [Jm_i]
            buff_JM = buff_JM + [JM_i]
            buff_J_joint = buff_Jm + buff_JM
            
            buff_X_batch = prev_X_batch.copy()
            buff_m_batch = prev_m_batch.copy()
            
            buff_X_batch = buff_X_batch + [Xq_i]
            buff_m_batch = buff_m_batch + [m_i]
            

            for m_j in range(self.M):
                
                Xq_j, info_j, Jm_j, JM_j = self.single_query_by_fidelity(
                    buff_X_batch, buff_m_batch, prev_acc_cost+self.costs[m_i], buff_Jm, buff_JM, buff_J_joint, m_j)
                
                reg_info_gain_mat[m_i, m_j] = (info_i+info_j)/(prev_acc_cost+self.costs[m_i]+self.costs[m_j])
                
                Xq_hist.append([Xq_i, Xq_j])
                Jm_hist.append([Jm_i, JM_j])
                JM_hist.append([JM_i, JM_j])
            #
        #

        
        (argm_i, argm_j) = np.unravel_index(np.argmax(reg_info_gain_mat), reg_info_gain_mat.shape)
        [Xq_i, Xq_j] = Xq_hist[np.argmax(reg_info_gain_mat)]
        [Jm_i, Jm_j] = Jm_hist[np.argmax(reg_info_gain_mat)]
        [JM_i, JM_j] = JM_hist[np.argmax(reg_info_gain_mat)]
        
        
        curr_X_batch = prev_X_batch + [Xq_i, Xq_j]
        curr_m_batch = prev_m_batch + [argm_i, argm_j]
        curr_J_m_batch = prev_J_m_batch + [Jm_i, Jm_j]
        curr_J_M_batch = prev_J_M_batch + [JM_i, JM_j]
        
        curr_acc_cost = prev_acc_cost + self.costs[argm_i] + self.costs[argm_j]
        
        curr_J_joint_batch = curr_J_m_batch + curr_J_M_batch
                
        
        return curr_X_batch, curr_m_batch, curr_acc_cost, curr_J_m_batch, curr_J_M_batch, curr_J_joint_batch

    def batch_query(self,):

        X_batch = []
        m_batch = []
        
        J_m_batch = []
        J_M_batch = []
        J_joint_batch = []
        
        acc_costs = 0.0
        
        for j in range(self.batch_size):
            
            if self.verbose:
                print('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***')
            self.logger.write('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***\n')
            self.logger.flush()

            X_batch, m_batch, acc_costs, J_m_batch, J_M_batch, J_joint_batch =\
                self.single_query(X_batch, m_batch, acc_costs, J_m_batch, J_M_batch, J_joint_batch)
            
        # for

        np_X_batch = []
        for xq in X_batch:
            np_X_batch.append(xq.data.cpu().numpy())
            
        if self.verbose:
            print(np_X_batch)
            print(m_batch)
            print(acc_costs)
            
        return np_X_batch, m_batch
    
    def batch_double_query(self):

        X_batch = []
        m_batch = []
        
        J_m_batch = []
        J_M_batch = []
        J_joint_batch = []
        
        acc_costs = 0.0
        
        for j in range(int(self.batch_size/2)):
            
            if self.verbose:
                print('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***')
            self.logger.write('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***\n')
            self.logger.flush()
            
            if self.verbose:
                print(len(J_m_batch))
                print(len(J_M_batch))
                print(len(J_joint_batch))
                print(m_batch)


            X_batch, m_batch, acc_costs, J_m_batch, J_M_batch, J_joint_batch =\
                self.double_query(X_batch, m_batch, acc_costs, J_m_batch, J_M_batch, J_joint_batch)

        # for

        np_X_batch = []
        for xq in X_batch:
            np_X_batch.append(xq.data.cpu().numpy())
            
        if self.verbose:
            print(np_X_batch)
            print(m_batch)
            print(acc_costs)
            
        return np_X_batch, m_batch
    
    def bound_eval_next_query(self, prev_X_batch, prev_m_batch, prev_J, prev_J_hf, i_batch, m):
        
        # replacing instead of concatenating
        
        max_retry = 10
        count = 0
        success = False
        
        Jm_batch = prev_J.copy()
        JM_batch = prev_J_hf.copy()
        X_batch = prev_X_batch.copy()
        m_batch = prev_m_batch.copy()
        
        
        while not success:
            if count <= max_retry:
                try:
                    Xq = self.init_query_points(m)

                    np_lb, np_ub = self.data.get_N_bounds(m)
                    bounds = torch.tensor(np.vstack((np_lb, np_ub)), device=self.device, dtype=self.torch_type)
                    
                    #print(bounds)
                        
                    lbfgs = LBFGS([Xq], self.opt_lr, line_search_fn='strong_wolfe')


                    Jm = self.eval_output_jacob(Xq, m)
                    JM = self.eval_output_jacob(Xq, self.M-1)
                    
                    Jm_batch[i_batch] = Jm
                    JM_batch[i_batch] = JM
                    J_joint = Jm_batch + JM_batch
                    
                    X_batch[i_batch] = Xq
                    m_batch[i_batch] = m

                    mutual_info = self.eval_batch_mutual_info(X_batch, m_batch, Jm_batch, JM_batch, J_joint)

                    if self.verbose:
                        print('Query m=%d info BEFORE Opt'%(m))
                        print('  - info:  ', mutual_info.data.cpu().numpy())
                        print('  - query: ', Xq.data.cpu().numpy())

                    self.logger.write("start to query fidelity m=" + str(m) + '\n')
                    self.logger.write("  - info BEFORE " + str(mutual_info.data.cpu().numpy()) + '\n')
                    self.logger.write("  - Xq   BEFORE " + str(Xq.data.cpu().numpy()) + '\n')


                    def closure():
                        lbfgs.zero_grad()  

                        Jm = self.eval_output_jacob(Xq, m)
                        JM = self.eval_output_jacob(Xq, self.M-1)

                        Jm_batch[i_batch] = Jm
                        JM_batch[i_batch] = JM
                        J_joint = Jm_batch + JM_batch

                        X_batch[i_batch] = Xq
                        m_batch[i_batch] = m

                        loss = -self.eval_batch_mutual_info(X_batch, m_batch, Jm_batch, JM_batch, J_joint)
                        #print(loss)
                        loss.backward(retain_graph=True)

                        with torch.no_grad():
                            for j, (lb, ub) in enumerate(zip(*bounds)):
                                Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                            #
                        #
                        
                        
                        return loss

                    lbfgs.step(closure)
                    
                    for j, (lb, ub) in enumerate(zip(*bounds)):
                        Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                    #


                    Jm = self.eval_output_jacob(Xq, m)
                    JM = self.eval_output_jacob(Xq, self.M-1)
                    
                    Jm_batch[i_batch] = Jm
                    JM_batch[i_batch] = JM
                    J_joint = Jm_batch + JM_batch
                    
                    X_batch[i_batch] = Xq
                    m_batch[i_batch] = m

                    mutual_info = self.eval_batch_mutual_info(X_batch, m_batch, Jm_batch, JM_batch, J_joint)
                    
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
                Xq = self.init_query_points(m)
                mutual_info = torch.tensor(0.0)
            #if
        # while
        
        return mutual_info, Xq
    
    def bound_single_query(self, prev_X_batch, prev_m_batch, prev_J_m_batch, prev_J_M_batch, i_batch):

        fidelity_info = []
        fidelity_query = []
        fidelity_costs = []
        
        prev_batch_costs = [self.costs[b] for b in prev_m_batch]

        for m in range(self.M):
            info, xq = self.bound_eval_next_query(
                prev_X_batch, prev_m_batch, prev_J_m_batch, prev_J_M_batch, i_batch, m)
            
            fidelity_info.append(info.data.cpu().numpy())
            fidelity_query.append(xq)
            
            copy_prev_batch_costs = prev_batch_costs.copy()
            copy_prev_batch_costs[i_batch] = self.costs[m]
            
            #print('origin',prev_batch_costs)
            #print('copy',copy_prev_batch_costs)
            
            fidelity_costs.append(np.sum(np.array(copy_prev_batch_costs)))

        #

        reg_info = np.array(fidelity_info) / np.array(fidelity_costs)
        
        #print(reg_info)
        
        argm = np.argmax(reg_info)
        argx = fidelity_query[argm]
        
        if self.verbose:
            print(argm)
            print(argx)
        self.logger.write('argm='+str(argm)+'\n')
        self.logger.write('argx='+str(argx.data.cpu().numpy())+'\n')
        self.logger.flush()
        
        Jm = self.eval_output_jacob(argx, argm)
        JM = self.eval_output_jacob(argx, self.M-1)
            
        curr_J_m_batch = prev_J_m_batch.copy()
        curr_J_M_batch = prev_J_M_batch.copy()
        
        curr_J_m_batch[i_batch] = Jm
        curr_J_M_batch[i_batch] = JM

        curr_X_batch = prev_X_batch.copy()
        curr_m_batch = prev_m_batch.copy()
        
        curr_X_batch[i_batch] = argx
        curr_m_batch[i_batch] = argm

        return curr_X_batch, curr_m_batch, curr_J_m_batch, curr_J_M_batch    

    
    def bound_double_query(self, prev_X_batch, prev_m_batch, prev_J_m_batch, prev_J_M_batch, i_batch, j_batch):

        info_gain_mat = np.zeros(shape=[self.M, self.M])
        costs_mat = np.zeros(shape=[self.M, self.M])
        
        hist_X_batch = []
        hist_m_batch = []
        hist_J_m_batch = []
        hist_J_M_batch = []
        
        prev_batch_costs = [self.costs[b] for b in prev_m_batch]

        for i in range(self.M):

            X_batch_i_copy = prev_X_batch.copy()
            m_batch_i_copy = prev_m_batch.copy()
            J_m_batch_i_copy = prev_J_m_batch.copy()
            J_M_batch_i_copy = prev_J_M_batch.copy()
        
            info_i, Xq_i = self.bound_eval_next_query(
                X_batch_i_copy, m_batch_i_copy, J_m_batch_i_copy, J_M_batch_i_copy, i_batch, i)
            
            Jm_i = self.eval_output_jacob(Xq_i, i)
            JM_i = self.eval_output_jacob(Xq_i, self.M-1)
            
#             print('## i-batch', i_batch)
#             print('##len batch', len(X_batch_i_copy))
#             print('##', len(prev_X_batch))
            
            
            X_batch_i_copy[i_batch] = Xq_i
            m_batch_i_copy[i_batch] = i
            J_m_batch_i_copy[i_batch] = Jm_i
            J_M_batch_i_copy[i_batch] = JM_i
            
            for j in range(self.M):
                
                X_batch_j_copy = X_batch_i_copy.copy()
                m_batch_j_copy = m_batch_i_copy.copy()
                J_m_batch_j_copy = J_m_batch_i_copy.copy()
                J_M_batch_j_copy = J_M_batch_i_copy.copy()
                
                info_j, Xq_j = self.bound_eval_next_query(
                X_batch_j_copy, m_batch_j_copy, J_m_batch_j_copy, J_M_batch_j_copy, j_batch, j)

                copy_prev_batch_costs = prev_batch_costs.copy()
                copy_prev_batch_costs[i_batch] = self.costs[i]
                copy_prev_batch_costs[j_batch] = self.costs[j]
                
                info_gain_mat[i, j] = info_i + info_j
                costs_mat[i, j] = np.sum(np.array(copy_prev_batch_costs))
                
                Jm_j = self.eval_output_jacob(Xq_j, j)
                JM_j = self.eval_output_jacob(Xq_j, self.M-1)

                X_batch_j_copy[j_batch] = Xq_j
                m_batch_j_copy[j_batch] = j
                J_m_batch_j_copy[j_batch] = Jm_j
                J_M_batch_j_copy[j_batch] = JM_j
                
                hist_X_batch.append(X_batch_j_copy.copy())
                hist_m_batch.append(m_batch_j_copy.copy())
                hist_J_m_batch.append(J_m_batch_j_copy.copy())
                hist_J_M_batch.append(J_M_batch_j_copy.copy())
                
            #
        #

        reg_info_gain_mat = info_gain_mat / costs_mat
        
        argi = np.argmax(reg_info_gain_mat)
        
#         print(hist_m_batch)
#         print(hist_m_batch[argi])

        return hist_X_batch[argi], hist_m_batch[argi], hist_J_m_batch[argi], hist_J_M_batch[argi]

    
#     def bound_batch_query(self,):

#         if self.M == 2:
#             p_init = [0.8, 0.2]
#         elif self.M == 3:
#             p_init = [0.8, 0.15, 0.05]
#         #
        
#         m_batch = np.random.choice(self.M, size=self.batch_size, p=p_init).tolist()
#         X_batch = []
#         for b in range(self.batch_size):
#             Xb = self.init_query_points(1)
#             X_batch.append(Xb)
#         #
        
#         #print(m_batch)
#         #print(X_batch)
        
#         J_m_batch = []
#         J_M_batch = []
        
#         for b in range(self.batch_size):
#             Jm = self.eval_output_jacob(X_batch[b], m_batch[b])
#             JM = self.eval_output_jacob(X_batch[b], self.M-1)
#             J_m_batch.append(Jm)
#             J_M_batch.append(JM)
#         #
        
#         for j in range(self.batch_size):
            
#             if self.verbose:
#                 print('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***')
#             self.logger.write('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***\n')
#             self.logger.flush()

#             X_batch, m_batch, J_m_batch, J_M_batch =\
#                 self.bound_single_query(X_batch, m_batch, J_m_batch, J_M_batch, j)
#             #
#         #
        
#         #print(m_batch)
#         #print(X_batch)

#         np_X_batch = []
#         for xq in X_batch:
#             np_X_batch.append(xq.data.cpu().numpy())
            
# #         if self.verbose:
# #             print(np_X_batch)
# #             print(m_batch)
            
#         return np_X_batch, m_batch
    
    def cycle_bound_batch_query(self, K=3):

        if self.M == 2:
            p_init = [0.8, 0.2]
        elif self.M == 3:
            p_init = [0.8, 0.15, 0.05]
        #
        
        m_batch = np.random.choice(self.M, size=self.batch_size, p=p_init).tolist()
        X_batch = []
        for b in range(self.batch_size):
            Xb = self.init_query_points(1)
            X_batch.append(Xb)
        #
        
        #print(m_batch)
        #print(X_batch)
        
        J_m_batch = []
        J_M_batch = []
        
        for b in range(self.batch_size):
            Jm = self.eval_output_jacob(X_batch[b], m_batch[b])
            JM = self.eval_output_jacob(X_batch[b], self.M-1)
            J_m_batch.append(Jm)
            J_M_batch.append(JM)
        #
        
        for k in range(K):
            for j in range(self.batch_size):

                if self.verbose:
                    print('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***')
                self.logger.write('*** query '+str(j+1)+'/'+str(self.batch_size) + ' sample ***\n')
                self.logger.flush()

                X_batch, m_batch, J_m_batch, J_M_batch =\
                    self.bound_single_query(X_batch, m_batch, J_m_batch, J_M_batch, j)
                #
            #
        #
        
        #print(m_batch)
        #print(X_batch)

        np_X_batch = []
        for xq in X_batch:
            np_X_batch.append(xq.data.cpu().numpy())
            
#         if self.verbose:
#             print(np_X_batch)
#             print(m_batch)
            
        return np_X_batch, m_batch
    
    def bound_batch_double_query(self,):

        if self.M == 2:
            p_init = [0.8, 0.2]
        elif self.M == 3:
            p_init = [0.8, 0.15, 0.05]
        #
        
        m_batch = np.random.choice(self.M, size=self.batch_size, p=p_init).tolist()
        X_batch = []
        for b in range(self.batch_size):
            Xb = self.init_query_points(1)
            X_batch.append(Xb)
        #
        
        #print(m_batch)
        #print(X_batch)
        
        J_m_batch = []
        J_M_batch = []
        
        for b in range(self.batch_size):
            Jm = self.eval_output_jacob(X_batch[b], m_batch[b])
            JM = self.eval_output_jacob(X_batch[b], self.M-1)
            J_m_batch.append(Jm)
            J_M_batch.append(JM)
        #
        
        _range = np.arange(0,self.batch_size,2).tolist()
        
        for j in _range:
            
            if self.verbose:
                print('*** query '+str(j+1)+'-'+str(j+2)+'/'+str(self.batch_size) + ' sample ***')
            self.logger.write('*** query '+str(j+1)+'-'+str(j+2)+'/'+str(self.batch_size) + ' sample ***\n')
            self.logger.flush()

            X_batch, m_batch, J_m_batch, J_M_batch =\
                self.bound_double_query(X_batch, m_batch, J_m_batch, J_M_batch, j, j+1)
            #
            
#             print('** X', len(X_batch))
#             print('** m_batch', len(m_batch))
#             print('** J_m_batch', len(J_m_batch))
#             print('** J_M_batch', len(J_M_batch))
            
        #
        
        #print(m_batch)
        #print(X_batch)

        np_X_batch = []
        for xq in X_batch:
            np_X_batch.append(xq.data.cpu().numpy())
            
#         if self.verbose:
#             print(np_X_batch)
#             print(m_batch)
            
        return np_X_batch, m_batch
    


    
#     def debug(self,):
#         print('debug mode')
        
# #         self.bound_batch_query()
# #         self.bound_batch_double_query()

# #         self.batch_query()
#         self.batch_double_query()

        
#         if self.M == 2:
#             p_init = [0.8, 0.2]
#         elif self.M == 3:
#             p_init = [0.8, 0.15, 0.05]
#         #
        
# # #         print(self.batch_size)
        
# #         init_m_batch = np.random.choice(self.M, size=self.batch_size, p=p_init).tolist()
# #         init_X_batch = []
# #         for b in range(self.batch_size):
# #             Xb = self.init_query_points(1)
# #             init_X_batch.append(Xb)
# #         #
        
# #         print(init_X_batch)
# #         print(init_m_batch)
        
# #         J_m_batch = []
# #         J_M_batch = []
        
# #         for b in range(self.batch_size):
# #             Jm = self.eval_output_jacob(init_X_batch[b], init_m_batch[b])
# #             JM = self.eval_output_jacob(init_X_batch[b], self.M-1)
# #             J_m_batch.append(Jm)
# #             J_M_batch.append(JM)
# #         #
        
# # #         i_batch = 1
# # #         m = 0
# # # #         self.bound_eval_next_query(init_X_batch, init_m_batch, J_m_batch, J_M_batch, i_batch, m)
# # #         self.bound_single_query(init_X_batch, init_m_batch, J_m_batch, J_M_batch, i_batch)


# #         i_batch = 1
# #         j_batch = 2
# #         m = 0
# # #         self.bound_eval_next_query(init_X_batch, init_m_batch, J_m_batch, J_M_batch, i_batch, m)
# #         self.bound_double_query(init_X_batch, init_m_batch, J_m_batch, J_M_batch, i_batch, j_batch)

# #         mi = self.eval_batch_mutual_info(init_X_batch, init_m_batch, J_m_batch, J_M_batch, J_m_batch+J_M_batch)
# #         print(mi)

        
# # # #         X_batch_curr, m_batch_curr, acc_cost_curr = self.single_query(X_batch, m_batch, 3)
        
# # # #         V_base = self.eval_batch_base_variance(X_batch_prev, m_batch_prev)
        
# # # #         print(V_base)

# # # #         self.eval_output_jacob(X,2)

# # #         J_batch = []
# # #         for i in range(len(m_batch_prev)):
# # #             J = self.eval_output_jacob(X_batch_prev[i], m_batch_prev[i])
# # #             J_batch.append(J)
            
# # #         V_base = self.eval_batch_base_variance_jacobians(X_batch_prev, m_batch_prev, J_batch)
        
# # #         print(V_base.data.cpu().numpy())

