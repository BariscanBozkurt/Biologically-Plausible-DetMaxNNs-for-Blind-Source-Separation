import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from numba import njit
from IPython.display import display, Latex, Math, clear_output
import pylab as pl
##### IMPORT MY UTILITY SCRIPTS #######
from BSSbase import *
from dsp_utils import *
from bss_utils import *
# from general_utils import *
from numba_utils import *
# from visualization_utils import * 

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

class OnlineBSM(BSSBaseClass):
    """
    BOUNDED SIMILARITY MATCHING
    Implementation of online one layer Weighted Bounded Source Seperation Recurrent Neural Network.
    Reference: Alper T. Erdoğan and Cengiz Pehlevan, 'Blind Source Seperation Using Neural Networks with Local Learning Rules',ICASSP 2020
    
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Initial guess for forward weight matrix W, must be size of s_dim by x_dim
    M              -- Initial guess for lateral weight matrix M, must be size of s_dim by s_dim
    D              -- Initial guess for weight (similarity weights) matrix, must be size of s_dim by s_dim
    gamma          -- Forgetting factor for data snapshot matrix
    mu, beta       -- Similarity weight update parameters, check equation (15) from the paper
    
    Methods:
    ==================================
    
    whiten_signal(X)        -- Whiten the given batch signal X

    fit_next_antisparse(x_online)     -- Updates the network parameters for one data point x_online
    
    fit_batch_antisparse(X_batch)     -- Updates the network parameters for given batch data X_batch (but in online manner)
    
    """
    def __init__(self, s_dim, x_dim, gamma = 0.9999, mu = 1e-3, beta = 1e-7, W = None, M = None, D = None, WScaling = 0.0033, 
                 GamScaling = 0.02, DScaling = 1, whiten_input_ = True, neural_OUTPUT_COMP_TOL = 1e-6, 
                 set_ground_truth = False, S = None, A = None):

        if W is not None:
            if whiten_input_:
                assert W.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d) (because of whitening)" % (s_dim, x_dim)
                W = W
            else:
                assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
                W = W
        else:
            if whiten_input_:
                W = np.random.randn(s_dim,s_dim)
                W = WScaling * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            else:
                W = np.random.randn(s_dim,x_dim)
                W = WScaling * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            
        if M is not None:
            assert M.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M
        else:
            M = GamScaling * np.eye(s_dim)  
            
        if D is not None:
            assert D.shape == (s_dim, 1), "The shape of the initial guess D must be (s_dim,1)=(%d,%d)" % (s_dim, 1)
            D = D
        else:
            D = DScaling * np.ones((s_dim,1))
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.gamma = gamma
        self.mu = mu
        self.beta = beta
        self.W = W
        self.M = M
        self.D = D
        self.whiten_input_ = whiten_input_
        self.Wpre = np.eye(x_dim)
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.set_ground_truth = set_ground_truth
        if set_ground_truth:
            self.S = S
            self.A = A
        else:
            self.S = None
            self.A = None

        self.SIR_list = []
        self.SNR_list = []

    def whiten_input(self, X):
        x_dim = self.x_dim
        s_dim = self.s_dim
        N = X.shape[1]
        # Mean of the mixtures
        mX = np.mean(X, axis = 1).reshape((x_dim, 1))
        # Covariance of Mixtures
        Rxx = np.dot(X, X.T)/N - np.dot(mX, mX.T)
        # Eigenvalue Decomposition
        d, V = np.linalg.eig(Rxx)
        D = np.diag(d)
        # Sorting indexis for eigenvalues from large to small
        ie = np.argsort(-d)
        # Inverse square root of eigenvalues
        ddinv = 1/np.sqrt(d[ie[:s_dim]])
        # Pre-whitening matrix
        Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)#*np.sqrt(12)
        # Whitened mixtures
        H = np.dot(Wpre, X)
        self.Wpre = Wpre
        return H, Wpre

    def evaluate_for_debug(self, W, A, S, X, mean_normalize_estimation = False):
        s_dim = self.s_dim
        Y_ = W @ X
        if mean_normalize_estimation:
            Y_ = Y_ - Y_.mean(axis = 1).reshape(-1,1)
        Y_ = self.signed_and_permutation_corrected_sources(S,Y_)
        coef_ = ((Y_ * S).sum(axis = 1) / (Y_ * Y_).sum(axis = 1)).reshape(-1,1)
        Y_ = coef_ * Y_

        SINR = 10*np.log10(self.CalculateSINRjit(Y_, S, False)[0])
        SNR = self.snr_jit(S, Y_)

        T = W @ A
        Tabs = np.abs(T)
        P = np.zeros((s_dim, s_dim))

        for SourceIndex in range(s_dim):
            Tmax = np.max(Tabs[SourceIndex,:])
            Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
            P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
        
        GG = P.T @ T
        _, SGG, _ = np.linalg.svd(GG) # SGG is the singular values of overall matrix Wf @ A

        return SINR, SNR, SGG, Y_, P

    def plot_for_debug(self, SIR_list, SNR_list, P, debug_iteration_point, YforPlot):
        pl.clf()
        pl.subplot(2,2,1)
        pl.plot(np.array(SIR_list), linewidth = 5)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.ylabel("SIR (dB)", fontsize = 45)
        pl.title("SIR Behaviour", fontsize = 45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2,2,2)
        pl.plot(np.array(SNR_list), linewidth = 5)
        pl.grid()
        pl.title("Component SNR Check", fontsize = 45)
        pl.ylabel("SNR (dB)", fontsize = 45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2,2,3)
        pl.plot(np.array(self.SV_list), linewidth = 5)
        pl.grid()
        pl.title("Singular Value Check, Overall Matrix Rank: " + str(np.linalg.matrix_rank(P)) , fontsize = 45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2,2,4)
        pl.plot(YforPlot, linewidth = 5)
        pl.title("Y last 25", fontsize = 45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())

    @staticmethod
    @njit
    def update_weights_jit(x_current, y, W, M, D, gamma, beta, mu):
        # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
        W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
        M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
        # D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
        D = (1 - beta) * D + mu * (np.sum(np.abs(W)**2,axis = 1) - np.sum((np.abs(M)**2) * D.T,axis=1)).reshape(-1,1)
        return W, M, D

    def compute_overall_mapping(self,return_mapping = True):
        W, M, D = self.W, self.M, self.D

        Wf = np.linalg.solve(M * D.T, W) @ self.Wpre
        # self.Wf = np.real(Wf)

        if return_mapping:
            return Wf
        else:
            return None

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, 
                                       tol = 1e-6, fast_start = False):

        def offdiag(A, return_diag = False):
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)

        M_hat, Upsilon = offdiag(M, True)

        u = Upsilon * ((D.T * y)[0])
        mat_factor1 = M_hat * D.T
        mat_factor2 = Upsilon * D.T
        if fast_start:
            u = 0.99*np.linalg.solve(M * D.T, W @ x)
            y = np.clip(u / mat_factor2[0], -1, 1)

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - mat_factor1 @ y)
            u = u - lr * du
            # y = y - lr * du

            y = np.clip(u / mat_factor2[0], -1, 1)

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):

        def offdiag(A, return_diag = False):
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)

        M_hat, Upsilon = offdiag(M, True)

        u = Upsilon * ((D.T * y)[0])
        mat_factor1 = M_hat * D.T
        mat_factor2 = Upsilon * D.T

        if fast_start:
            u = 0.99*np.linalg.solve(M * D.T, W @ x)
            y = np.clip(u / mat_factor2[0], 0, 1)
            # y = np.clip(np.linalg.solve(M * D.T, W @ x), 0, 1)
        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - mat_factor1 @ y)
            # u = u - lr * du
            y = y - lr * du

            y = np.clip(u / mat_factor2[0], 0, 1)

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    def fit_next_antisparse(self, x_current, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False):
        W = self.W
        M = self.M
        D = self.D
        gamma, mu, beta = self.gamma, self.mu, self.beta
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL

        y = np.random.randn(self.s_dim,)
        y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)

        
        W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
        M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
        
        D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
        
        self.W = W
        self.M = M
        self.D = D
        
    def fit_batch_antisparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, 
                             neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):

        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)
        else:
            S = None
            A = None

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]

        Y = 0.05*np.random.randn(self.s_dim, samples)
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_input(X)
            X_white = np.real(X_white)
            self.A = A
        else:
            X_white = X
            

        for k in range(n_epochs):
            for i_sample in tqdm(range(samples)):
                x_current = X_white[:, idx[i_sample]] # Take one input
                y = Y[:, idx[i_sample]]

                # Upsilon = np.diag(np.diag(M)) # Following paragraph of equation (16)
                
                # Neural Dynamics: Equations (17) from the paper
                y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, 
                                                        neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, 
                                                        fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                W, M, D = self.update_weights_jit(x_current, y, W, M, D, gamma, beta, mu)
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            self.W = W
                            self.M = M
                            self.D = D

                            Wf = self.compute_overall_mapping(return_mapping = True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(Wf, A, S, X, False)
                            self.SV_list.append(abs(SGG))

                            SIR_list.append(SINR)
                            SNR_list.append(SNR)

                            self.SNR_list = SNR_list
                            self.SIR_list = SIR_list

                            if plot_in_jupyter:
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                        except Exception as e:
                            print(str(e))
        self.W = W
        self.M = M
        self.D = D

    def fit_batch_nnantisparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)
        else:
            S = None
            A = None

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]

        Y = np.zeros((self.s_dim, samples))
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_input(X)
            X_white = np.real(X_white)
            self.A = A
        else:
            X_white = X
            
        for k in range(n_epochs):
            for i_sample in tqdm(range(samples)):
                x_current = X_white[:, idx[i_sample]] # Take one input
                y = Y[:, idx[i_sample]]

                # Upsilon = np.diag(np.diag(M)) # Following paragraph of equation (16)
                
                # Neural Dynamics: Equations (17) from the paper
                y = self.run_neural_dynamics_nnantisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, 
                                                          neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                W, M, D = self.update_weights_jit(x_current, y, W, M, D, gamma, beta, mu)
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            self.W = W
                            self.M = M
                            self.D = D

                            Wf = self.compute_overall_mapping(return_mapping = True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(Wf, A, S, X, False)
                            self.SV_list.append(abs(SGG))

                            SIR_list.append(SINR)
                            SNR_list.append(SNR)

                            self.SNR_list = SNR_list
                            self.SIR_list = SIR_list

                            if plot_in_jupyter:
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                        except Exception as e:
                            print(str(e))
        self.W = W
        self.M = M
        self.D = D
