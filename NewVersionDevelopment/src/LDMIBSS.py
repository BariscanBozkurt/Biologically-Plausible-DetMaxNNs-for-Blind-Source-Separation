import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from numba import njit
from IPython.display import display, Latex, Math, clear_output
import pylab as pl
##### IMPORT MY UTILITY SCRIPTS #######
from dsp_utils import *
from bss_utils import *
# from general_utils import *
from numba_utils import *
# from visualization_utils import * 

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

class LDMIBSS:

    """
    Implementation of batch Log-Det Mutual Information Based Blind Source Separation Framework
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Feedforward Synapses

    Methods:
    ==================================
    fit_batch_antisparse
    fit_batch_nnantisparse

    """
    
    def __init__(self, s_dim, x_dim, W = None, set_ground_truth = False, S = None, A = None):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.random.randn(s_dim, x_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W = W # Trainable separator matrix, i.e., W@X \approx S
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []

    def CalculateSIR(self, H,pH, return_db = True):
        G=pH@H
        Gmax=np.diag(np.max(abs(G),axis=1))
        P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
        T=G@P.T
        rankP=np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT,2)**2
        # Interference Power
        intpow = np.linalg.norm(T,'fro')**2 - sigpow
        SIRV = sigpow/intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10*np.log10(sigpow/intpow)

        return SIRV,rankP

    @staticmethod
    @njit( parallel=True )
    def snr_jit(S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 1)
        S_P = (S_original ** 2).sum(axis = 1)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    @staticmethod
    @njit
    def CalculateSINRjit(Out,S, compute_permutation = True):
        def mean_numba(a):
            res = []
            for i in range(a.shape[0]):
                res.append(a[i, :].mean())

            return np.array(res)
        
        r=S.shape[0]
        Smean = mean_numba(S)
        Outmean = mean_numba(Out)
        if compute_permutation:
            G=np.dot(Out-np.reshape(Outmean,(r,1)),np.linalg.pinv(S-np.reshape(Smean,(r,1))))
            #G = np.linalg.lstsq((S-np.reshape(Smean,(r,1))).T, (Out-np.reshape(Outmean,(r,1))).T)[0]
            indmax = np.abs(G).argmax(1).astype(np.int64)
        else:
            G=np.dot(Out-np.reshape(Outmean,(r,1)),np.linalg.pinv(S-np.reshape(Smean,(r,1))))
            #G = np.linalg.lstsq((S-np.reshape(Smean,(r,1))).T, (Out-np.reshape(Outmean,(r,1))).T)[0]
            indmax = np.arange(0,r)

        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:] - Outmean[kk], S[indmax[kk],:].T - Smean[indmax[kk]])/np.dot(S[indmax[kk],:] - Smean[indmax[kk]], S[indmax[kk],:].T - Smean[indmax[kk]])#(G[kk,indmax[kk]])

        ZZ = GG @ (S-np.reshape(Smean,(r,1))) + np.reshape(Outmean,(r,1))
        E = Out - ZZ
        MSE = np.linalg.norm(E)**2
        SigPow = np.linalg.norm(ZZ)**2
        SINR = (SigPow/MSE)
        return SINR,SigPow,MSE,G

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S,Y):
        """
        S    : Original source matrix
        Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
        
        return the permutation of the source seperation algorithm
        """
        # perm = np.argmax(np.abs(self.outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
        perm = np.argmax(np.abs(outer_prod_broadcasting(Y.T,S.T).sum(axis = 0))/(np.linalg.norm(S,axis = 1)*np.linalg.norm(Y,axis=1)), axis = 0)
        return perm

    def signed_and_permutation_corrected_sources(self,S,Y):
        """_summary_

        Args:
            S (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return (np.sign((Y[perm,:] * S).sum(axis = 1))[:,np.newaxis]) * Y[perm,:]

    def evaluate_for_debug(self, W, Y, A, S, X):
        s_dim = self.s_dim
        # Y_ = W @ X
        Y_ = self.signed_and_permutation_corrected_sources(S,Y)
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
        pl.title("Random 25 Output (from Y)", fontsize = 45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())
       
    @staticmethod
    @njit
    def update_Y_corr_based(Y, X, W, epsilon, step_size):
        s_dim, samples = Y.shape[0], Y.shape[1]
        Identity_like_Y = np.eye(s_dim)
        RY = (1/samples) * np.dot(Y, Y.T) + epsilon * Identity_like_Y
        E = Y - np.dot(W, X)
        RE = (1/samples) * np.dot(E, E.T) + epsilon * Identity_like_Y
        gradY = (1/samples) * (np.dot(np.linalg.pinv(RY), Y) - np.dot(np.linalg.pinv(RE), E))
        Y = Y + (step_size) * gradY
        return Y

    @staticmethod
    @njit
    def update_Y_cov_based(Y, X, muX, W, epsilon, step_size):
        def mean_numba(a):

            res = []
            for i in range(a.shape[0]):
                res.append(a[i, :].mean())

            return np.array(res)
        s_dim, samples = Y.shape[0], Y.shape[1]
        muY = mean_numba(Y).reshape(-1,1)
        Identity_like_Y = np.eye(s_dim)
        RY = (1/samples) * (np.dot(Y, Y.T) - np.dot(muY, muY.T)) + epsilon * Identity_like_Y
        E = (Y - muY) - np.dot(W, (X - muX.reshape(-1,1)))
        muE = mean_numba(E).reshape(-1,1)
        RE = (1/samples) * (np.dot(E, E.T) - np.dot(muE, muE.T)) + epsilon * Identity_like_Y
        gradY = (1/samples) * (np.dot(np.linalg.pinv(RY), Y - muY) - np.dot(np.linalg.pinv(RE), E - muE))
        Y = Y + (step_size) * gradY
        return Y

    @staticmethod
    @njit
    def ProjectOntoLInfty(X):
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    @staticmethod
    @njit
    def ProjectOntoNNLInfty(X):
        return X*(X>=0.0)*(X<=1.0)+(X>1.0)*1.0#-0.0*(X<0.0)
        
    def ProjectRowstoL1NormBall(self, H):
        Hshape=H.shape
        #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
        lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
        #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

        u=-np.sort(-np.abs(H),axis=1)
        sv=np.cumsum(u,axis=1)
        q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
        rho=np.max(q,axis=1)
        rho=rho.astype(int)
        lindex=np.linspace(1,Hshape[0],Hshape[0])-1
        lindex=lindex.astype(int)
        theta=np.maximum(0,np.reshape((sv[tuple([lindex,rho])]-1)/(rho+1),(Hshape[0],1)))
        ww=np.abs(H)-theta
        H=np.sign(H)*(ww>0)*ww
        return H

    def ProjectColstoSimplex(self, v, z=1):
        """v array of shape (n_features, n_samples)."""
        p, n = v.shape
        u = np.sort(v, axis=0)[::-1, ...]
        pi = np.cumsum(u, axis=0) - z
        ind = (np.arange(p) + 1).reshape(-1, 1)
        mask = (u - pi / ind) > 0
        rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
        theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w

    def fit_batch_antisparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = (np.random.rand(self.s_dim, samples) - 0.5)/2
        for k in tqdm(range(n_iterations)):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoLInfty(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoLInfty(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    try:
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, Y, A, S, X)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(Y.shape[1]-25)
                            YforPlot = Y[:,random_idx-25:random_idx].T
                            self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                    except Exception as e:
                        print(str(e))
        self.W = W

    def fit_batch_nnantisparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        Y = np.random.rand(self.s_dim, samples)/2
        for k in tqdm(range(n_iterations)):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoNNLInfty(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoNNLInfty(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    try:
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, Y, A, S, X)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(Y.shape[1]-25)
                            YforPlot = Y[:,random_idx-25:random_idx].T
                            self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                    except Exception as e:
                        print(str(e))
        
    def fit_batch_sparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in tqdm(range(n_iterations)):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall(Y.T).T
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall(Y.T).T
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    try:
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, Y, A, S, X)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(Y.shape[1]-25)
                            YforPlot = Y[:,random_idx-25:random_idx].T
                            self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                    except Exception as e:
                        print(str(e))
        self.W = W

    def fit_batch_nnsparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in tqdm(range(n_iterations)):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall((Y * (Y>= 0)).T).T
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall((Y * (Y>= 0)).T).T
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    try:
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, Y, A, S, X)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(Y.shape[1]-25)
                            YforPlot = Y[:,random_idx-25:random_idx].T
                            self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                    except Exception as e:
                        print(str(e))  
        self.W = W

    def fit_batch_simplex(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (45, 30), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in tqdm(range(n_iterations)):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectColstoSimplex(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectColstoSimplex(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    try:
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, Y, A, S, X)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(Y.shape[1]-25)
                            YforPlot = Y[:,random_idx-25:random_idx].T
                            self.plot_for_debug(SIR_list, SNR_list, P, debug_iteration_point, YforPlot)
                    except Exception as e:
                        print(str(e))  
        self.W = W
