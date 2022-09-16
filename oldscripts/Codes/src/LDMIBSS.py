from random import sample
import numpy as np
import scipy
from scipy.stats import invgamma, chi2, t
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib as mpl
import math
import pylab as pl
from numba import njit, jit
from tqdm import tqdm
from IPython.display import display, Latex, Math, clear_output
from IPython import display as display1
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull  
from numpy.linalg import det
from scipy.stats import dirichlet
import itertools
import pypoman
from utils import *
import warnings
warnings.filterwarnings("ignore")

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
        self.SINR_list = []
        self.SNR_list = []

    # Calculate SIR Function
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
        """
        S    : Original source matrix
        Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
        
        return the permutation of the source seperation algorithm
        """
        # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
        # perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
        # perm = np.argmax(np.abs(outer_prod_broadcasting(S,Y).sum(axis = 0)), axis = 0)
        perm = np.argmax(np.abs(self.outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
        return perm

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

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

    # @njit(parallel=True)
    # def mean_numba(a):

    #     res = []
    #     for i in range(a.shape[0]):
    #         res.append(a[i, :].mean())

    #     return np.array(res)

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
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = (np.random.rand(self.s_dim, samples) - 0.5)/2
        for k in range(n_iterations):
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
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    # SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    # SIRlist.append(SIR)
                    # self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        self.W = W

    def fit_batch_nnantisparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)

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
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    # SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    # SIRlist.append(SIR)
                    # self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        
    def fit_batch_sparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in range(n_iterations):
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
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    # SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    # SIRlist.append(SIR)
                    # self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        self.W = W

    def fit_batch_nnsparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in range(n_iterations):
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
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    # SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    # SIRlist.append(SIR)
                    # self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())   
        self.W = W

    def fit_batch_simplex(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in range(n_iterations):
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
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    # SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    # SIRlist.append(SIR)
                    # self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())   
        self.W = W

class MinibatchLDMIBSS:

    """
    Implementation of batch Log-Det Mutual Information Based Blind Source Separation Framework

    ALMOST THE SAME IMPLEMENTATION WITH THE ABOVE LDMIBSS CLASS. THE ONLY DIFFERENCE IS THAT 
    THIS ALGORITHM UPDATES ARE PERFORMED BASED ON THE MINIBATCHES. THE ABOVE LDMIBSS CLASS IS 
    WORKING SLOW WHEN THE NUMBER OF DATA IS BIG (FOR EXAMPLE THE MIXTURE SIZE IS (Nmixtures, Nsamples) = (10, 500000)).
    THEREFORE, IN EACH ITERATION, WE TAKE A MINIBATCH OF MIXTURES TO RUN THE ALGORITHM. IN THE DEBUGGING
    PART FOR SNR ANS SINR CALCULATION, THE WHOLE DATA IS USED (NOT THE MINIBATCHES).

    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Feedforward Synapses
    By             -- Inverse Output Covariance
    Be             -- Inverse Error Covariance
    lambday        -- Ry forgetting factor
    lambdae        -- Re forgetting factor

    
    Methods:
    ==================================
    run_neural_dynamics_antisparse
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
        self.SINR_list = []
        self.SNR_list = []

    # Calculate SIR Function
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
        """
        S    : Original source matrix
        Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
        
        return the permutation of the source seperation algorithm
        """
        # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
        # perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
        # perm = np.argmax(np.abs(outer_prod_broadcasting(S,Y).sum(axis = 0)), axis = 0)
        perm = np.argmax(np.abs(self.outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
        return perm

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

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

    # @njit(parallel=True)
    # def mean_numba(a):

    #     res = []
    #     for i in range(a.shape[0]):
    #         res.append(a[i, :].mean())

    #     return np.array(res)

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

    def ProjectOntoLInfty(self, X):
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def ProjectOntoNNLInfty(self, X):
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

    def fit_batch_antisparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.random.randn(self.s_dim, Xbatch.shape[1])/5
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectOntoLInfty(Ybatch)
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectOntoLInfty(Ybatch)
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                        # if (((k % debug_iteration_point) == 0) | (k == n_iterations_per_batch - 1)) & (k >= debug_iteration_point):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            # SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            # SIRlist.append(SIR)
                            # self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

    def fit_batch_nnantisparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.random.rand(self.s_dim, Xbatch.shape[1])/2
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectOntoNNLInfty(Ybatch)
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectOntoNNLInfty(Ybatch)
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                        # if (((k % debug_iteration_point) == 0) | (k == n_iterations_per_batch - 1)) & (k >= debug_iteration_point):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            # SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            # SIRlist.append(SIR)
                            # self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  
        
    def fit_batch_sparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectRowstoL1NormBall(Ybatch.T).T
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectRowstoL1NormBall(Ybatch.T).T
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            # SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            # SIRlist.append(SIR)
                            # self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

    def fit_batch_nnsparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectRowstoL1NormBall((Ybatch * (Ybatch >= 0)).T).T
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectRowstoL1NormBall((Ybatch * (Ybatch >= 0)).T).T
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            # SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            # SIRlist.append(SIR)
                            # self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

    def fit_batch_simplex(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectColstoSimplex(Ybatch)
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectColstoSimplex(Ybatch)
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            # SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            # SIRlist.append(SIR)
                            # self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                # pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

######### FOLLOWING FUNCTIONS ARE MOVED TO utils.py
# def whiten_signal(X, mean_normalize = True, type_ = 3):
#     """
#     Input : X  ---> Input signal to be whitened
    
#     type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
    
#     Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
#     """
#     if mean_normalize:
#         X = X - np.mean(X,axis = 0, keepdims = True)
    
#     cov = np.cov(X.T)
    
#     if type_ == 3: # Whitening using singular value decomposition
#         U,S,V = np.linalg.svd(cov)
#         d = np.diag(1.0 / np.sqrt(S))
#         W_pre = np.dot(U, np.dot(d, U.T))
        
#     else: # Whitening using eigenvalue decomposition
#         d,S = np.linalg.eigh(cov)
#         D = np.diag(d)

#         D_sqrt = np.sqrt(D * (D>0))

#         if type_ == 1: # Type defines how you want W_pre matrix to be
#             W_pre = np.linalg.pinv(S@D_sqrt)
#         elif type_ == 2:
#             W_pre = np.linalg.pinv(S@D_sqrt@S.T)
    
#     X_white = (W_pre @ X.T).T
    
#     return X_white, W_pre

# def whiten_input(X, n_components = None, return_prewhitening_matrix = False):
#     """
#     X.shape[0] = Number of sources
#     X.shape[1] = Number of samples for each signal
#     """
#     x_dim = X.shape[0]
#     if n_components is None:
#         n_components = x_dim
#     s_dim = n_components
    
#     N = X.shape[1]
#     # Mean of the mixtures
#     mX = np.mean(X, axis = 1).reshape((x_dim, 1))
#     # Covariance of Mixtures
#     Rxx = np.dot(X, X.T)/N - np.dot(mX, mX.T)
#     # Eigenvalue Decomposition
#     d, V = np.linalg.eig(Rxx)
#     D = np.diag(d)
#     # Sorting indexis for eigenvalues from large to small
#     ie = np.argsort(-d)
#     # Inverse square root of eigenvalues
#     ddinv = 1/np.sqrt(d[ie[:s_dim]])
#     # Pre-whitening matrix
#     Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)#*np.sqrt(12)
#     # Whitened mixtures
#     H = np.dot(Wpre, X)
#     if return_prewhitening_matrix:
#         return H, Wpre
#     else:
#         return H

# def ZeroOneNormalizeData(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))

# def ZeroOneNormalizeColumns(X):
#     X_normalized = np.empty_like(X)
#     for i in range(X.shape[1]):
#         X_normalized[:,i] = ZeroOneNormalizeData(X[:,i])

#     return X_normalized

# def Subplot_gray_images(I, image_shape = [512,512], height = 15, width = 15, title = ''):
#     n_images = I.shape[1]
#     fig, ax = plt.subplots(1,n_images)
#     fig.suptitle(title)
#     fig.set_figheight(height)
#     fig.set_figwidth(width)
#     for i in range(n_images):
#         ax[i].imshow(I[:,i].reshape(image_shape[0],image_shape[1]), cmap = 'gray')
    
#     plt.show()

# def subplot_1D_signals(X, title = '',title_fontsize = 20, figsize = (10,5), linewidth = 1, colorcode = '#050C12'):
#     """
#     Plot the 1D signals (each column from the given matrix)
#     """
#     n = X.shape[1] # Number of signals
    
#     fig, ax = plt.subplots(n,1, figsize = figsize)
    
#     for i in range(n):
#         ax[i].plot(X[:,i], linewidth = linewidth, color = colorcode)
#         ax[i].grid()
    
#     plt.suptitle(title, fontsize = title_fontsize)
#     # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
#     # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
#     plt.draw()

# def plot_convergence_plot(metric, xlabel = '', ylabel = '', title = '', figsize = (12,8), fontsize = 15, linewidth = 3, colorcode = '#050C12'):
    
#     plt.figure(figsize = figsize)
#     plt.plot(metric, linewidth = linewidth, color = colorcode)
#     plt.xlabel(xlabel, fontsize = fontsize)
#     plt.ylabel(ylabel, fontsize = fontsize)
#     plt.title(title, fontsize = fontsize)
#     # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
#     # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
#     plt.grid()
#     plt.draw()
    
# def find_permutation_between_source_and_estimation(S,Y):
#     """
#     S    : Original source matrix
#     Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
    
#     return the permutation of the source seperation algorithm
#     """
    
#     # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
#     perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
#     return perm

# def signed_and_permutation_corrected_sources(S,Y):
#     perm = find_permutation_between_source_and_estimation(S,Y)
#     return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

# def psnr(img1, img2, pixel_max = 1):
#     """
#     Return peak-signal-to-noise-ratio between given two images
#     """
#     mse = np.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     else:
#         return 20 * np.log10(pixel_max / np.sqrt(mse))

# def snr(S_original, S_noisy):
#     N_hat = S_original - S_noisy
#     N_P = (N_hat ** 2).sum(axis = 0)
#     S_P = (S_original ** 2).sum(axis = 0)
#     snr = 10 * np.log10(S_P / N_P)
#     return snr

# def ProjectOntoLInfty(X, thresh = 1.0):
#     return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

# def ProjectOntoNNLInfty(X, thresh = 1.0):
#     return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh

# def ProjectRowstoL1NormBall(H):
#     Hshape=H.shape
#     #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
#     lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
#     #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

#     u=-np.sort(-np.abs(H),axis=1)
#     sv=np.cumsum(u,axis=1)
#     q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
#     rho=np.max(q,axis=1)
#     rho=rho.astype(int)
#     lindex=np.linspace(1,Hshape[0],Hshape[0])-1
#     lindex=lindex.astype(int)
#     theta=np.maximum(0,np.reshape((sv[tuple([lindex,rho])]-1)/(rho+1),(Hshape[0],1)))
#     ww=np.abs(H)-theta
#     H=np.sign(H)*(ww>0)*ww
#     return H

# def ProjectColstoSimplex(v, z=1):
#     """v array of shape (n_features, n_samples)."""
#     p, n = v.shape
#     u = np.sort(v, axis=0)[::-1, ...]
#     pi = np.cumsum(u, axis=0) - z
#     ind = (np.arange(p) + 1).reshape(-1, 1)
#     mask = (u - pi / ind) > 0
#     rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
#     theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
#     w = np.maximum(v - theta, 0)
#     return w

# def projection_simplex(V, z=1, axis=None):
#     """
#     Projection of x onto the simplex, scaled by z:
#         P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
#     z: float or array
#         If array, len(z) must be compatible with V
#     axis: None or int
#         axis=None: project V by P(V.ravel(); z)
#         axis=1: project each V[i] by P(V[i]; z[i])
#         axis=0: project each V[:, j] by P(V[:, j]; z[j])
#     """
#     if axis == 1:
#         n_features = V.shape[1]
#         U = np.sort(V, axis=1)[:, ::-1]
#         z = np.ones(len(V)) * z
#         cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
#         ind = np.arange(n_features) + 1
#         cond = U - cssv / ind > 0
#         rho = np.count_nonzero(cond, axis=1)
#         theta = cssv[np.arange(len(V)), rho - 1] / rho
#         return np.maximum(V - theta[:, np.newaxis], 0)

#     elif axis == 0:
#         return projection_simplex(V.T, z, axis=1).T

#     else:
#         V = V.ravel().reshape(1, -1)
#         return projection_simplex(V, z, axis=1).ravel()
       
# def display_matrix(array):
#     data = ''
#     for line in array:
#         if len(line) == 1:
#             data += ' %.3f &' % line + r' \\\n'
#             continue
#         for element in line:
#             data += ' %.3f &' % element
#         data += r' \\' + '\n'
#     display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))

# # Calculate SIR Function
# def CalculateSIR(H,pH, return_db = True):
#     G=pH@H
#     Gmax=np.diag(np.max(abs(G),axis=1))
#     P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
#     T=G@P.T
#     rankP=np.linalg.matrix_rank(P)
#     diagT = np.diag(T)
#     # Signal Power
#     sigpow = np.linalg.norm(diagT,2)**2
#     # Interference Power
#     intpow = np.linalg.norm(T,'fro')**2 - sigpow
#     SIRV = sigpow/intpow
#     # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
#     if return_db:
#         SIRV = 10*np.log10(sigpow/intpow)

#     return SIRV,rankP

# def CalculateSINR(Out, S, compute_permutation = True):
#     r=S.shape[0]
#     if compute_permutation:
#         G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
#         indmax=np.argmax(np.abs(G),1)
#     else:
#         G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
#         indmax = np.arange(0,r)
#     GG=np.zeros((r,r))
#     for kk in range(r):
#         GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
#     ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
#     E=Out-ZZ
#     MSE=np.linalg.norm(E,'fro')**2
#     SigPow=np.linalg.norm(ZZ,'fro')**2
#     SINR=(SigPow/MSE)
#     return SINR,SigPow,MSE,G

# @njit(fastmath = True)
# def accumu(lis):
#     """
#     Cumulative Sum. Same as np.cumsum()
#     """
#     result = np.zeros_like(lis)
#     for i in range(lis.shape[1]):
#         result[:,i] = np.sum(lis[:,:i+1])

#     return result

# @njit(fastmath = True)
# def merge_sort(list_):
#     """
#     Sorts a list in ascending order.
#     Returns a new sorted list.
    
#     Divide : Find the midpoint of the list and divide into sublist
#     Conquer : Recursively sort the sublists created in previous step
#     Combine : Merge the sorted sublists created in previous step
    
#     Takes O(n log n) time.
#     """
    
#     def merge(left, right):
#         """
#         Merges two lists (arrays), sorting them in the process.
#         Returns a new merged list
        
#         Runs in overall O(n) time
#         """
        
#         l = []
#         i = 0
#         j = 0
        
#         while i < len(left) and j < len(right):
#             if left[i] < right[j]:
#                 l.append(left[i])
#                 i += 1
#             else:
#                 l.append(right[j])
#                 j += 1
                
#         while i < len(left):
#             l.append(left[i])
#             i += 1
            
#         while j < len(right):
#             l.append(right[j])
#             j += 1
        
#         return l

#     def split(list_):
#         """
#         Divide the unsorted list at midpoint into sublists.
#         Returns two sublists - left and right
        
#         Takes overall O(log n) time
#         """
        
#         mid = len(list_) // 2
        
#         left = list_[:mid]
#         right = list_[mid:]
        
#         return left, right

#     if len(list_) <= 1:
#         return list_
    
#     left_half, right_half = split(list_)
#     left = merge_sort(left_half)
#     right = merge_sort(right_half)
    
#     return np.array(merge(left, right))

# # @njit
# # def ProjectVectortoL1NormBall(H):
# #     Hshape = H.shape
# #     lr = np.repeat(np.reshape((1/np.linspace(1, H.shape[1], H.shape[1]))))

# def addWGN(signal, SNR, return_noise = False, print_resulting_SNR = False):
#     """
#     Adding white Gaussian Noise to the input signal
#     signal              : Input signal, numpy array of shape (number of sources, number of samples)
#                           If your signal is a 1D numpy array of shape (number of samples, ), then reshape it 
#                           by signal.reshape(1,-1) before giving it as input to this function
#     SNR                 : Desired input signal to noise ratio
#     print_resulting_SNR : If you want to print the numerically calculated SNR, pass it as True
    
#     Returns
#     ============================
#     signal_noisy        : Output signal which is the sum of input signal and additive noise
#     noise               : Returns the added noise
#     """
#     sigpow = np.mean(signal**2, axis = 1)
#     noisepow = 10 **(-SNR/10) * sigpow
#     noise =  np.sqrt(noisepow)[:,np.newaxis] * np.random.randn(signal.shape[0], signal.shape[1])
#     signal_noisy = signal + noise
#     if print_resulting_SNR:
#         SNRinp = 10 * np.log10(np.sum(np.mean(signal**2, axis = 1)) / np.sum(np.mean(noise**2, axis = 1)))
#         print("Input SNR is : {}".format(SNRinp))
#     if return_noise:
#         return signal_noisy, noise
#     else:
#         return signal_noisy

# def WSM_Mixing_Scenario(S, NumberofMixtures = None, INPUT_STD = None):
#     NumberofSources = S.shape[0]
#     if INPUT_STD is None:
#         INPUT_STD = S.std()
#     if NumberofMixtures is None:
#         NumberofMixtures = NumberofSources
#     A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
#     X = A @ S
#     for M in range(A.shape[0]):
#         stdx = np.std(X[M,:])
#         A[M,:] = A[M,:]/stdx * INPUT_STD
        
#     return A, X

# def generate_correlated_uniform_sources(R, range_ = [-1,1], n_sources = 5, size_sources = 500000):
#     """
#     R : correlation matrix
#     """
#     assert R.shape[0] == n_sources, "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)" % (n_sources)
#     S = np.random.uniform(range_[0], range_[1], size = (n_sources, size_sources))
#     L = np.linalg.cholesky(R)
#     S_ = L @ S
#     return S_

# def generate_correlated_copula_sources(rho = 0.0, df = 4, n_sources = 5, size_sources = 500000, decreasing_correlation = True):
#     """
#     rho     : correlation parameter
#     df      : degrees for freedom

#     required libraries:
#     from scipy.stats import invgamma, chi2, t
#     from scipy import linalg
#     import numpy as np
#     """
#     if decreasing_correlation:
#         first_row = np.array([rho ** j for j in range(n_sources)])
#         calib_correl_matrix = linalg.toeplitz(first_row, first_row)
#     else:
#         calib_correl_matrix = np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho

#     mu = np.zeros(len(calib_correl_matrix))
#     s = chi2.rvs(df, size = size_sources)[:, np.newaxis]
#     Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
#     X = np.sqrt(df/s) * Z # chi-square method
#     S = t.cdf(X, df).T
#     return S

# def generate_uniform_points_in_polytope(polytope_vertices, size):
#     """"
#     polytope_vertices : vertex matrix of shape (n_dim, n_vertices)

#     return:
#         Samples of shape (n_dim, size)
#     """
#     polytope_vertices = polytope_vertices.T
#     dims = polytope_vertices.shape[-1]
#     hull = polytope_vertices[ConvexHull(polytope_vertices).vertices]
#     deln = hull[Delaunay(hull).simplices]

#     vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
#     sample = np.random.choice(len(vols), size = size, p = vols / vols.sum())

#     return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = size)).T

# def generate_practical_polytope(dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list):
#     A = []
#     b = []
#     for j in antisparse_dims:
#         row1 = [0 for _ in range(dim)]
#         row2 = row1.copy()
#         row1[j] = 1
#         A.append(row1)
#         b.append(1)
#         row2[j] = -1
#         A.append(row2)
#         b.append(1)

#     for j in nonnegative_dims:
#         row1 = [0 for _ in range(dim)]
#         row2 = row1.copy()
#         row1[j] = 1
#         A.append(row1)
#         b.append(1)
#         row2[j] = -1
#         A.append(row2)
#         b.append(0)

#     for relative_sparse_dims in relative_sparse_dims_list:
#         row = np.zeros(dim)
#         pm_one = [[1,-1] for _ in range(relative_sparse_dims.shape[0])]
#         for i in itertools.product(*pm_one):
#             row_copy = row.copy()
#             row_copy[relative_sparse_dims] = i
#             A.append(list(row_copy))
#             b.append(1)
#     A = np.array(A)
#     b = np.array(b)
#     vertices = pypoman.compute_polytope_vertices(A, b)
#     V = np.array([list(v) for v in vertices]).T
#     return (A,b), V
