import numpy as np
import scipy
import math
from scipy.stats import invgamma, chi2, t
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib as mpl
from tqdm import tqdm
from numba import njit, jit
import mne
from mne.preprocessing import ICA
import logging
from time import time
import os
from IPython.display import display, Latex, Math, clear_output
import pylab as pl
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull  
from numpy.linalg import det
from scipy.stats import dirichlet
import itertools
import pypoman
# from utils import *
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15


@njit
def clipping(inp,lev):
    out=inp*(np.abs(inp)<=lev)+lev*(inp>lev)-lev*(inp<-lev)
    return out

################ WEIGHTED SIMILARITY MATCHING BLIND SOURCE SEPARATION #######################
class OnlineWSMBSS:

    def __init__(self, s_dim, x_dim, h_dim = None, gamma_start = 0.2, gamma_stop = 0.001, beta = 0.5, zeta = 1e-4, muD = [25,25], 
                 W_HX = None, W_YH = None, M_H = None, M_Y = None, D1 = None, D2 = None, WScalings = [0.0033,0.0033], 
                 GamScalings = [0.02, 0.02], DScalings = [25,1], LayerMinimumGains = [1e-6,1], LayerMaximumGains = [1e6,1], 
                 neural_OUTPUT_COMP_TOL = 1e-5, set_ground_truth = False, S = None, A = None ):

        if h_dim is None:
            h_dim = s_dim
        else:
            h_dim = h_dim
        
        if W_HX is not None:
            assert W_HX.shape == (h_dim, x_dim), "The shape of the initial guess W must be (h_dim,x_dim)=(%d,%d)" % (h_dim, x_dim)
            W_HX = W_HX
        else:
            W_HX = np.random.standard_normal(size = (h_dim, x_dim))
            for k in range(W_HX.shape[0]):
                W_HX[k,:] = WScalings[0] * W_HX[k,:]/np.linalg.norm(W_HX[k,:])

        if W_YH is not None:
            assert W_YH.shape == (s_dim, h_dim), "The shape of the initial guess W must be (s_dim,h_dim)=(%d,%d)" % (s_dim, h_dim)
            W_YH = W_YH
        else:
            W_YH = np.random.standard_normal(size = (s_dim, h_dim))
            for k in range(W_YH.shape[0]):
                W_YH[k,:] = WScalings[1] * W_YH[k,:]/np.linalg.norm(W_YH[k,:])

        if M_H is not None:
            assert M_H.shape == (h_dim, h_dim), "The shape of the initial guess M must be (h_dim,h_dim)=(%d,%d)" % (h_dim, h_dim)
            M_H = M_H
        else:
            M_H = GamScalings[0] * np.eye(h_dim)   

        if M_Y is not None:
            assert M_Y.shape == (s_dim, s_dim), "The shape of the initial guess M must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M_Y = M_Y
        else:
            M_Y = GamScalings[1] * np.eye(s_dim)

        if D1 is not None:
            assert D1.shape == (h_dim, h_dim), "The shape of the initial guess D must be (h_dim,h_dim)=(%d,%d)" % (h_dim, h_dim)
            D1 = D1
        else:
            D1 = DScalings[0] * np.eye(h_dim)

        if D2 is not None:
            assert D2.shape == (s_dim, s_dim), "The shape of the initial guess D must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            D2 = D2
        else:
            D2 = DScalings[1] * np.eye(s_dim)

        self.s_dim = s_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.gamma_start = gamma_start
        self.gamma_stop = gamma_stop
        self.beta = beta
        self.zeta = zeta
        self.muD = muD
        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.LayerMinimumGains = LayerMinimumGains
        self.LayerMaximumGains = LayerMaximumGains
        self.Y = None
        self.H = None
        self.WScalings = WScalings
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []

    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR SYNAPSE & GAIN UPDATES ETC. #########################
    ############################################################################################
    def ProjectOntoLInfty(self, X, thresh):
        return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

    def sthreshold(self,x, thresh):
        absolute = np.abs(x)
        sign = np.sign(x)
        return (absolute>thresh) * (absolute - thresh) * sign

    def bsthreshold(self,x, thresh):
        absolute = np.abs(x)
        sign = np.sign(x)
        return (absolute > thresh) * (absolute - thresh) * sign * (absolute < (1 + thresh)) + sign * (absolute >= (1+thresh))

    def dlogdet(self, D, DEPS = 5e-8):
        d = np.diag(np.diag(D + DEPS * np.eye(len(D))) ** (-1))
        return d

    def d2logdet(self, D, DEPS = 5e-8):
        d2 = -np.diag(np.diag(D + DEPS*np.eye(len(D)))**(-2))
        return d2

    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ###########################################
    ############################################################################################
    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def ZeroOneNormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def ZeroOneNormalizeColumns(self,X):
        X_normalized = np.empty_like(X)
        for i in range(X.shape[1]):
            X_normalized[:,i] = self.ZeroOneNormalizeData(X[:,i])

        return X_normalized

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S,Y):
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

    def signed_and_permutation_corrected_sources(self,S,Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                           neural_dynamic_iterations, lr_start, lr_stop, OUTPUT_COMP_TOL):

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y

        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            # delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            # delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            # delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            # v = v + MUV * delv

            # h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))
            # h = h*(h>=-2.0)*(h<=2.0)+(h>2.0)*2.0-2.0*(h<-2.0)

            # delu = -u + (1 - zeta) * (1 - beta) * D2 @ W_YH @ h
            # delu = delu - (1 - zeta) * (1 - beta) * D2 @ M_hat_Y @ D2 @ y
            # u = u + MUV * delu

            # y = u / np.diag(Gamma_Y * ((1 - zeta) * (1 - beta) * D2 **2))
            # y = y*(y>=-1.0)*(y<=1.0)+(y>1.0)*1.0-1.0*(y<-1.0)


            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + W_YH @ h
            delu = delu - M_hat_Y @ D2 @ y
            u = u + (MUV) * delu
            y = u / np.diag(Gamma_Y * (D2))
            y = y*(y>=-1.0)*(y<=1.0)+(y>1.0)*1.0-1.0*(y<-1.0)


            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u
            
        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse_jit(x_current, h, y, M_H, M_Y, 
                                            W_HX, W_YH, D1, D2, beta, zeta, neural_dynamic_iterations, 
                                            lr_start, lr_stop, OUTPUT_COMP_TOL,
                                            use_adam_opt = False, adam_beta1 = 0.9, adam_beta2 = 0.999, adam_eps = 1e-8):

        if use_adam_opt:
            beta1 = adam_beta1
            beta2 = adam_beta2
            epsilon = adam_eps
            mt_h = np.zeros_like(h)
            vt_h = np.zeros_like(h)
            mt_y = np.zeros_like(y)
            vt_y = np.zeros_like(y)

            Gamma_H = np.diag(np.diag(M_H))
            M_hat_H = M_H - Gamma_H

            Gamma_Y = np.diag(np.diag(M_Y))
            M_hat_Y = M_Y - Gamma_Y

            v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ (h)
            u = Gamma_Y @ D2 @ (y)

            PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
            MembraneVoltageNotSettled = 1
            OutputCounter = 0

            while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
                OutputCounter += 1
                MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

                delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
                delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
                delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
                
                mt_h = beta1 * mt_h + (1 - beta1)*delv
                vt_h = beta2 * vt_h + (1 - beta2)*delv**2

                mt_hat_h = mt_h / (1 - beta1**OutputCounter)
                vt_hat_h = vt_h / (1 - beta2**OutputCounter)

                v = v + MUV * mt_hat_h / (np.sqrt(vt_hat_h) + epsilon)

                h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

                delu = -u + W_YH @ h
                delu = delu - M_hat_Y @ D2 @ y

                mt_y = beta1 * mt_y + (1 - beta1)*delu
                vt_y = beta2 * vt_y + (1 - beta2)*delu**2

                mt_hat_y = mt_y / (1 - beta1**OutputCounter)
                vt_hat_y = vt_y / (1 - beta2**OutputCounter)
                u = u + MUV * mt_hat_y / (np.sqrt(vt_hat_y) + epsilon)

                y = u / np.diag(Gamma_Y * (D2))
                y = y*(y>=0)*(y<=1.0)+(y>1.0)*1.0

                MembraneVoltageNotSettled = 0
                if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/(np.linalg.norm(v) + 1e-30) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/(np.linalg.norm(u) + 1e-30) > OUTPUT_COMP_TOL):
                    MembraneVoltageNotSettled = 1
                PreviousMembraneVoltages['v'] = v
                PreviousMembraneVoltages['u'] = u
        else:
            Gamma_H = np.diag(np.diag(M_H))
            M_hat_H = M_H - Gamma_H

            Gamma_Y = np.diag(np.diag(M_Y))
            M_hat_Y = M_Y - Gamma_Y

            v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ (h)
            u = Gamma_Y @ D2 @ (y)

            PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
            MembraneVoltageNotSettled = 1
            OutputCounter = 0

            while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
                OutputCounter += 1
                MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

                delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
                delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
                delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
                v = v + (MUV) * delv
                h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

                delu = -u + W_YH @ h
                delu = delu - M_hat_Y @ D2 @ y
                u = u + (MUV) * delu
                y = u / np.diag(Gamma_Y * (D2))
                y = y*(y>=0)*(y<=1.0)+(y>1.0)*1.0
                v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ (h)
                u = Gamma_Y @ D2 @ (y)
                MembraneVoltageNotSettled = 0
                if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/(np.linalg.norm(v) + 1e-30) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/(np.linalg.norm(u) + 1e-30) > OUTPUT_COMP_TOL):
                    MembraneVoltageNotSettled = 1
                PreviousMembraneVoltages['v'] = v
                PreviousMembraneVoltages['u'] = u
                
        return h,y, OutputCounter
    
    @staticmethod
    @njit
    def run_neural_dynamics_mixedantisparse_jit(x_current, h, y, nn_components, signed_components, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                                neural_dynamic_iterations, lr_start, lr_stop, OUTPUT_COMP_TOL):

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y

        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv

            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + W_YH @ h
            delu = delu - M_hat_Y @ D2 @ y
            u = u + (MUV) * delu
            y = u / np.diag(Gamma_Y * (D2))
            y[signed_components] = y[signed_components]*(y[signed_components]>=-1.0)*(y[signed_components]<=1.0)+(y[signed_components]>1.0)*1.0-1.0*(y[signed_components]<-1.0)
            y[nn_components] = y[nn_components] *(y[nn_components] >=0)*(y[nn_components] <=1.0)+(y[nn_components] >1.0)*1.0

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u
            
        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_sparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                       neural_dynamic_iterations, lr_start, lr_stop = 0.05, OUTPUT_COMP_TOL = 1e-6):

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y
        
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        STLAMBD = 0
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + (1 - zeta) * (1 - beta) * D2 @ W_YH @ h
            delu = delu - ((1 - zeta) * (1 - beta) * D2 @ M_hat_Y @ D2) @ y
            u = u + MUV * delu
            y = u / np.diag(Gamma_Y * ((1 - zeta) * (1 - beta) * D2 ** 2))

            y_absolute = np.abs(y)
            y_sign = np.sign(y)
            y = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign
            y = y*(y>=-1.0)*(y<=1.0)+(y>1.0)*1.0-1.0*(y<-1.0)

            dval = np.linalg.norm(y,1) - 1
            
            STLAMBD = max(STLAMBD + 1 * dval,0)
            
            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u     

        return h,y

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                       neural_dynamic_iterations, lr_start, lr_stop = 0.05, OUTPUT_COMP_TOL = 1e-6):

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y
        
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        STLAMBD = 0
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            # delu = -u + (1 - zeta) * (1 - beta) * D2 @ W_YH @ h
            # delu = delu - ((1 - zeta) * (1 - beta) * D2 @ M_hat_Y @ D2) @ y
            # u = u + MUV * delu
            # y = u / np.diag(Gamma_Y * ((1 - zeta) * (1 - beta) * D2 ** 2))
            delu = -u + W_YH @ h
            delu = delu - (M_hat_Y @ D2) @ y
            u = u + MUV * delu
            y = u / np.diag(Gamma_Y * (D2))

            y = np.maximum(y - STLAMBD, 0)
            dval = np.sum(y) - 1
            STLAMBD = max(STLAMBD + 0.2 * dval, 0)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u     

        return h,y

    @staticmethod
    @njit
    def run_neural_dynamics_simplex_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                        neural_dynamic_iterations, lr_start, lr_stop = 0.05, OUTPUT_COMP_TOL = 1e-6):

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y
        
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        STLAMBD = 0
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            delv = -(1 - zeta) * v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))
            h = h*(h>=-2.0)*(h<=2.0)+(h>2.0)*2.0-2.0*(h<-2.0)

            # delu = -u + (1 - zeta) * (1 - beta) * D2 @ W_YH @ h
            # delu = delu - ((1 - zeta) * (1 - beta) * D2 @ M_hat_Y @ D2) @ y
            # u = u + MUV * delu
            # y = u / np.diag(Gamma_Y * ((1 - zeta) * (1 - beta) * D2 ** 2))

            delu = -u + W_YH @ h
            delu = delu - (M_hat_Y @ D2) @ y
            u = u + MUV * delu
            y = u / np.diag(Gamma_Y * (D2))

            y = np.maximum(y - STLAMBD, 0)
            dval = np.sum(y) - 1
            STLAMBD = STLAMBD + 0.01 * dval

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u     

        return h,y

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubsparse_jit(x_current, h, y, nn_components, sparse_components, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                                neural_dynamic_iterations, lr_start, lr_stop, OUTPUT_COMP_TOL):
        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y
        STLAMBD = 0
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv

            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + W_YH @ h
            delu = delu - M_hat_Y @ D2 @ y
            u = u + (MUV) * delu
            y = u / np.diag(Gamma_Y * (D2))
            y[nn_components] = y[nn_components]*(y[nn_components]>=0)*(y[nn_components]<=1.0)+(y[nn_components]>1.0)*1.0


            y_sparse_absolute = np.abs(y[sparse_components])
            y_sparse_sign = np.sign(y[sparse_components])
            y[sparse_components] = (y_sparse_absolute > STLAMBD) * (y_sparse_absolute - STLAMBD) * y_sparse_sign
            y = y*(y>=-1.0)*(y<=1.0)+(y>1.0)*1.0-1.0*(y<-1.0)

            dval = np.linalg.norm(y[sparse_components],1) - 1
            
            STLAMBD = max(STLAMBD + 1.5 * dval,0)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u
            
        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubnnsparse_jit(x_current, h, y, nn_components, nnsparse_components, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                                neural_dynamic_iterations, lr_start, lr_stop, OUTPUT_COMP_TOL):

        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh #-thresh*(X<-thresh)
        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y
        STLAMBD = 0
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.5), lr_stop)

            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv

            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + W_YH @ h
            delu = delu - M_hat_Y @ D2 @ y
            u = u + (MUV) * delu
            y = u / np.diag(Gamma_Y * (D2))
            y[nn_components] = ProjectOntoNNLInfty(y[nn_components])
            
            y[nnsparse_components] = np.maximum(y[nnsparse_components] - STLAMBD, 0)
            dval = np.sum(y[nnsparse_components]) - 1
            STLAMBD = max(STLAMBD + 0.2 * dval, 0)
            y = ProjectOntoNNLInfty(y)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u
            
        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_olhaussen_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, neural_dynamic_iterations, neural_lr, OUTPUT_COMP_TOL):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

        def sthreshold(x, thresh = 0):
            absolute = np.abs(x)
            sign = np.sign(x)
            return (absolute>thresh) * (absolute - thresh) * sign

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y

        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0 
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            STLAMBD = 0
            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y

            v = v + neural_lr * delv / np.sqrt(OutputCounter)
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + zeta * (1 - beta) * D2 @ W_YH @ h
            delu = delu - (zeta * (1 - beta) * D2 @ M_hat_Y @ D2) @ y
            u = u + neural_lr * delu / np.sqrt(OutputCounter)
            a = u / np.diag(Gamma_Y * (zeta * (1 - beta) * D2 ** 2))

            y = sthreshold(a,0)

            temp = 1
            if np.linalg.norm(a,1) >= 1:
                iter2 = 0

                while ((np.abs(STLAMBD - temp) / np.abs(STLAMBD + 1e-10)) > 1e-5) & (iter2 < 10):

                    iter2 += 1
                    temp = STLAMBD
                    y = sthreshold(a, STLAMBD)

                    sstep = 2e-2 / np.sqrt(iter2)
                    dval = np.linalg.norm(y,1) - 1
                    STLAMBD = STLAMBD + sstep * dval
                    if STLAMBD < 0:
                        STLAMBD = 0
                        y = a
                y = ProjectOntoLInfty(y,1)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u  
        
        return h,y,OutputCounter  

    @staticmethod
    @njit
    def run_neural_dynamics_general_polytope_jit(x_current, h, y, signed_dims, nn_dims, sparse_dims_list, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                                neural_dynamic_iterations, lr_start, lr_stop = 0.05, lambda_lr = 1, OUTPUT_COMP_TOL = 1e-6):

        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh
        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded
        def ReLU(X):
            return np.maximum(X,0)

        def loop_intersection(lst1, lst2):
            result = []
            for element1 in lst1:
                for element2 in lst2:
                    if element1 == element2:
                        result.append(element1)
            return result

        STLAMBD_list = np.zeros(len(sparse_dims_list))

        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y

        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)

            delv = -v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))

            delu = -u + W_YH @ h
            delu = delu - M_hat_Y @ D2 @ y
            u = u + (MUV) * delu
            y = u / np.diag(Gamma_Y * (D2))
            
            if sparse_dims_list[0][0] != -1:
                for ss,sparse_dim in enumerate(sparse_dims_list):
                    # y[sparse_dim] = SoftThresholding(y[sparse_dim], STLAMBD_list[ss])
                    # STLAMBD_list[ss] = max(STLAMBD_list[ss] + (np.linalg.norm(y[sparse_dim],1) - 1), 0)
                    if signed_dims[0] != -1:
                        y[np.array(loop_intersection(sparse_dim, signed_dims))] = SoftThresholding(y[np.array(loop_intersection(sparse_dim, signed_dims))], STLAMBD_list[ss])
                    if nn_dims[0] != -1:
                        y[np.array(loop_intersection(sparse_dim, nn_dims))] = ReLU(y[np.array(loop_intersection(sparse_dim, nn_dims))] - STLAMBD_list[ss])
                    STLAMBD_list[ss] = max(STLAMBD_list[ss] + lambda_lr * (np.linalg.norm(y[sparse_dim],1) - 1), 0)
            if signed_dims[0] != -1:
                y[signed_dims] = ProjectOntoLInfty(y[signed_dims])
            if nn_dims[0] != -1:
                y[nn_dims] = ProjectOntoNNLInfty(y[nn_dims])

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u
            
        return h,y, OutputCounter

    ###############################################################
    ############## WSMBSS ALGORITHMS ##############################
    ###############################################################

    def compute_overall_mapping(self, return_mapping = False):
        beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH = self.beta, self.zeta, self.D1, self.D2, self.M_H, self.M_Y, self.W_HX, self.W_YH
        # Mapping from xt -> ht
        WL1 = np.linalg.inv((1 -zeta) * beta * D1 @ M_H @ D1 + (1 - zeta) * (1 - beta) * M_H - (1 -zeta) * (1 - beta) * W_YH.T @ np.linalg.inv(M_Y) @ W_YH) @ ((1 - zeta) * beta * D1 @ W_HX)

        # Mapping from ht -> yt
        WL2 = np.linalg.inv(D2) @ np.linalg.inv(M_Y) @ W_YH

        try: 
            W_pre = self.W_pre
        except:
            W_pre = np.eye(self.x_dim)
        
        # Seperator
        W = WL2 @ WL1 @ W_pre
        
        self.W = W 

        if return_mapping:
            return W

    def predict(self, X):
        W = self.compute_overall_mapping(return_mapping=True)

        return W @ X

    def fit_batch_antisparse(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        gamma_start, gamma_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.gamma_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1minlist = []
        D2minlist = []
        self.SV_list = []
        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]

        if self.Y is None:
            H = np.zeros((h_dim,samples))
            Y = np.zeros((s_dim,samples))
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                
                if ((i_sample + 1) % 100000) == 0:
                    muD = 0.99 * np.array(muD)
                
                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]
                neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL

                h,y, _ = self.run_neural_dynamics_antisparse_jit(x_current = x_current, h = h, y = y, 
                                                                M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                                D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                                OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1+np.log(2+ i_sample)),gamma_stop])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)

                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)

                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)

                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)
                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ = np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_

                            SNR_list.append(self.snr(S.T,Y_))
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                            if plot_in_jupyter:
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)

                                # pl.subplot(2,2,1)
                                # pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Diagonal Values of D1", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,2)
                                # pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Diagonal Values of D2", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,3)
                                # pl.plot(np.array(SNR_list), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.ylabel("SNR (dB)", fontsize = 45)
                                # pl.title("Component SNR Check", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,4)
                                # pl.plot(np.array(self.SV_list), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # clear_output(wait=True)
                                # display(pl.gcf())   
                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.plot(np.array(D1maxlist))
                                pl.grid()
                                # pl.legend(["D1min", "D1max"])
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.plot(np.array(D2maxlist))
                                pl.grid()
                                # pl.legend(["D2min","D2max"])
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())  

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                        except Exception as e:
                            print(str(e))
        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list

    def fit_batch_nnantisparse(self, X, n_epochs = 1, neural_dynamic_iterations = 750, use_adam_opt = False, adam_opt_params = {'beta1':0.9, 'beta2':0.99, 'eps': 1e-8},neural_lr_start = 0.2, neural_lr_stop = 0.05, neural_fast_start = False, whiten = False, shuffle = True, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        gamma_start, gamma_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.gamma_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1minlist = []
        D2minlist = []
        self.SV_list = []
        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]

        if self.Y is None:
            H = np.zeros((h_dim,samples))
            Y = np.zeros((s_dim,samples))

        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                
                if ((i_sample + 1) % 100000) == 0:
                    muD = 0.99 * np.array(muD)

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]] 

                h = H[:,idx[i_sample]]

                h,y, oc = self.run_neural_dynamics_nnantisparse_jit(x_current = x_current, h = h, 
                                                                    y = y, M_H = M_H, M_Y = M_Y,
                                                                    W_HX = W_HX, W_YH = W_YH, 
                                                                    D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                    neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                    lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                                    OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL, 
                                                                    use_adam_opt = use_adam_opt, adam_beta1 = adam_opt_params['beta1'],
                                                                    adam_beta2 = adam_opt_params['beta2'], adam_eps = adam_opt_params['eps'])

                MUS = np.max([gamma_start/(1+ np.log(1 + i_sample)/10),gamma_stop])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)

                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)

                D1_prev = D1.copy()
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                # D1secderivative = (1 - zeta) * beta * np.diag((np.diag(M_H)**2) * np.diag(D1)) + zeta * self.d2logdet(D1)
                # D1 = D1 - muD[0] * D1derivative #/ D1secderivative
                D1 = D1 - clipping(muD[0] * D1derivative, D1 * 1)

                D2_prev = D2.copy()
                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                # D2secderivative = (1 - zeta) * beta * np.diag((np.diag(M_Y) ** 2) * np.diag(D2)) + zeta * self.d2logdet(D2)
                # D2 = D2 - muD[1] * D2derivative #/ D2secderivative
                D2 = D2 - clipping(muD[1] * D2derivative, D2 * 1) 

                d1 = np.diag(D1)
                d2 = np.diag(D2)

                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if np.min(d1 - LayerMinimumGains[0]) < 0.1:
                    D1 = D1_prev
                if np.min(d2 - LayerMinimumGains[1]) < 0.1:
                    D2 = D2_prev

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:

                            W = self.compute_overall_mapping(return_mapping = True)
                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999


                            perm = self.find_permutation_between_source_and_estimation(Y.T, S.T)
                            
                            GG = P.T @ T
                            _, SGG, _=np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            # diagGG = np.diag(GG)

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_

                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)

                                # pl.subplot(2,2,1)
                                # pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Diagonal Values of D1", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,2)
                                # pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Diagonal Values of D2", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,3)
                                # pl.plot(np.array(SNR_list), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.ylabel("SNR (dB)", fontsize = 45)
                                # pl.title("Component SNR Check", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,4)
                                # pl.plot(np.array(self.SV_list), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)


                                # clear_output(wait=True)
                                # display(pl.gcf())  
                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.plot(np.array(D1maxlist))
                                pl.grid()
                                # pl.legend(["D1min", "D1max"])
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.plot(np.array(D2maxlist))
                                pl.grid()
                                # pl.legend(["D2min","D2max"])
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                        except Exception as e:
                            print(str(e))

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list

    def fit_batch_mixedantisparse(self, X, nn_components, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        gamma_start, gamma_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.gamma_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1minlist = []
        D2minlist = []
        self.SV_list = []
        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        
        source_indices = [j for j in range(s_dim)]
        signed_components = source_indices.copy()
        for a in nn_components:
            signed_components.remove(a)
        nn_components = np.array(nn_components)
        signed_components = np.array(signed_components)

        if self.Y is None:
            H = np.zeros((h_dim,samples))
            Y = np.zeros((s_dim,samples))
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                
                if ((i_sample + 1) % 100000) == 0:
                    muD = 0.99 * np.array(muD)
                
                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]
                neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL

                h,y, _ = self.run_neural_dynamics_mixedantisparse_jit(x_current = x_current, h = h, y = y, 
                                                                        nn_components = nn_components,
                                                                        signed_components = signed_components,
                                                                        M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                                        D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                        neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                        lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                                        OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1+np.log(2+ i_sample)),gamma_stop])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)

                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)

                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)

                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)
                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ = np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_

                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))

                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d2_min = np.diag(D1), np.diag(D2)
                                # D1minlist.append(d1_min)
                                # D2minlist.append(d2_min)

                                # pl.subplot(2,2,1)
                                # pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Diagonal Values of D1", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,2)
                                # pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Diagonal Values of D2", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,3)
                                # pl.plot(np.array(SNR_list), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.ylabel("SNR (dB)", fontsize = 45)
                                # pl.title("Component SNR Check", fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # pl.subplot(2,2,4)
                                # pl.plot(np.array(self.SV_list), linewidth = 5)
                                # pl.grid()
                                # pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                # pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                # pl.xticks(fontsize=45)
                                # pl.yticks(fontsize=45)

                                # clear_output(wait=True)
                                # display(pl.gcf())   
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.plot(np.array(D1maxlist))
                                pl.grid()
                                # pl.legend(["D1min", "D1max"])
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.plot(np.array(D2maxlist))
                                pl.grid()
                                # pl.legend(["D2min","D2max"])
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                        except Exception as e:
                            print(str(e))
        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list

    def fit_batch_sparse(self, X, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []

        if self.Y is None:
            H = np.zeros((h_dim,samples))
            Y = np.zeros((s_dim,samples))
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y = self.run_neural_dynamics_sparse_jit(x_current = x_current, h = h, y = y, M_H = M_H, M_Y = M_Y, 
                                                          W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                          neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                          lr_stop = neural_lr_stop, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            diagGG = np.diag(GG)
                            # Signal Power
                            sigpow = np.linalg.norm(diagGG,2)**2
                            # Interference Power
                            intpow = np.linalg.norm(GG, 'fro')**2 - sigpow

                            SIR = 10*np.log10(sigpow/intpow)

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_
                            
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))

                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                # pl.plot(np.array(D1maxlist))
                                pl.grid()
                                # pl.legend(["D1min", "D1max"])
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                # pl.plot(np.array(D2maxlist))
                                pl.grid()
                                # pl.legend(["D2min","D2max"])
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list               

    def fit_batch_nnsparse(self, X, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []

        if self.Y is None:
            H = np.zeros((h_dim,samples)) + 0.001
            Y = np.zeros((s_dim,samples)) + 0.001
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y = self.run_neural_dynamics_nnsparse_jit(x_current = x_current, h = h, y = y, M_H = M_H, M_Y = M_Y, 
                                                          W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                          neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                          lr_stop = neural_lr_stop, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            Y_ = W @ X
                            Yzeromean = Y_ - Y_.mean(axis = 1).reshape(-1,1)
                            Y_ = self.signed_and_permutation_corrected_sources(Szeromean.T,Yzeromean.T)
                            coef_ = (Y_ * Szeromean.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_ 
                            
                            # SIR_list.append(SIR)
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, Szeromean, False)[0]))

                            SNR_list.append(self.snr(Szeromean.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list               

    def fit_batch_simplex(self, X, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []

        if self.Y is None:
            H = np.zeros((h_dim,samples)) 
            Y = np.zeros((s_dim,samples))
            # H = np.random.randn(h_dim, samples)
            # Y = np.random.randn(h_dim, samples)
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y = self.run_neural_dynamics_simplex_jit(x_current = x_current, h = h, y = y, M_H = M_H, M_Y = M_Y, 
                                                          W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                          neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                          lr_stop = neural_lr_stop, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative
                # D1 = D1 - clipping(muD[0] * D1derivative, D1 * 1)

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative
                # D2 = D2 - clipping(muD[1] * D2derivative, D2 * 1)

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            Y_ = W @ X
                            Yzeromean = Y_ - Y_.mean(axis = 1).reshape(-1,1)
                            Y_ = self.signed_and_permutation_corrected_sources(Szeromean.T,Yzeromean.T)
                            coef_ = (Y_ * Szeromean.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_ 
                            
                            # SIR_list.append(SIR)
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, Szeromean, False)[0]))

                            SNR_list.append(self.snr(Szeromean.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list               
           
    def fit_batch_nnwsubsparse(self, X, sparse_components, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []

        if self.Y is None:
            H = np.zeros((h_dim,samples)) 
            Y = np.zeros((s_dim,samples)) 
        else:
            H, Y = self.H, self.Y

        source_indices = [j for j in range(self.s_dim)]
        nn_components = source_indices.copy()
        for a in sparse_components:
            nn_components.remove(a)
        sparse_components = np.array(sparse_components)
        nn_components = np.array(nn_components)

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y, _ = self.run_neural_dynamics_nnwsubsparse_jit(x_current = x_current, h = h, y = y, 
                                                          nn_components = nn_components, sparse_components = sparse_components,
                                                          M_H = M_H, M_Y = M_Y, 
                                                          W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                          neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                          lr_stop = neural_lr_stop, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            self.Y_ = Y_
                            
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))

                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list               

    def fit_batch_nnwsubnnsparse(self, X, nnsparse_components, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []

        if self.Y is None:
            H = np.zeros((h_dim,samples)) 
            Y = np.zeros((s_dim,samples)) 
        else:
            H, Y = self.H, self.Y

        source_indices = [j for j in range(self.s_dim)]
        nn_components = source_indices.copy()
        for a in nnsparse_components:
            nn_components.remove(a)
        nnsparse_components = np.array(nnsparse_components)
        nn_components = np.array(nn_components)

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y, _ = self.run_neural_dynamics_nnwsubnnsparse_jit(x_current = x_current, h = h, y = y, 
                                                          nn_components = nn_components, nnsparse_components = nnsparse_components,
                                                          M_H = M_H, M_Y = M_Y, 
                                                          W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                          neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                          lr_stop = neural_lr_stop, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_ * (Y_ >= 0)
                            self.Y_ = Y_
                            
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))

                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list               
      
    def fit_batch_olhaussen(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr = 5, shuffle = True, debug_iteration_point = 1000, plot_rf = False, default_start = True):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains

        if default_start:
            W_HX = np.eye(self.h_dim,self.x_dim) + 0.01 * np.random.standard_normal(size = (self.h_dim, self.x_dim))
            W_YH = np.eye(self.s_dim, self.h_dim) + 0.01 * np.random.standard_normal(size = (self.s_dim, self.h_dim))

            for k in range(W_HX.shape[0]):
                W_HX[k,:] = self.WScalings[0] * W_HX[k,:]/np.linalg.norm(W_HX[k,:])
            for k in range(W_YH.shape[0]):
                W_YH[k,:] = self.WScalings[1] * W_YH[k,:]/np.linalg.norm(W_YH[k,:])


        self.W_HX = W_HX
        self.W_YH = W_YH       

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        x_dim = self.x_dim

        if self.Y is None:
            H = np.zeros((h_dim,samples))
            Y = np.zeros((s_dim,samples))
        else:
            H, Y = self.H, self.Y

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]
                neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL

                h,y,oc = self.run_neural_dynamics_olhaussen_jit(x_current = x_current, h = h, y = y, M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, neural_dynamic_iterations = neural_dynamic_iterations, neural_lr = neural_lr, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = gamma_start

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)

                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)

                D1derivative = (1 - zeta) * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                D1 = D1 - muD[0] * D1derivative

                D2derivative = (1 - zeta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                D2 = D2 - muD[1] * D2derivative

                d1 = np.diag(D1)
                d2 = np.diag(D2)

                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if (i_sample % debug_iteration_point) == 0:        
                    try:   
                        # Mapping from xt -> ht
                        WL1 = np.linalg.inv((1 -zeta) * beta * D1 @ M_H @ D1 + (1 - zeta) * (1 - beta) * M_H - (1 -zeta) * (1 - beta) * W_YH.T @ np.linalg.inv(M_Y) @ W_YH) @ ((1 - zeta) * beta * D1 @ W_HX)

                        # Mapping from ht -> yt
                        WL2 = np.linalg.inv(D2) @ np.linalg.inv(M_Y) @ W_YH

                        # Seperator
                        W = WL2 @ WL1

                        self.W = W

                        if plot_rf:
                            n_iterations = k*samples + i_sample
                            print('The receptive fields after {}'.format(n_iterations))
                            
                            fig, ax = plt.subplots(12,12, figsize = (20,20))

                            for l in range(144):
                                rf = np.reshape(W[l,:], (12,12))
                                rf = ZeroOneNormalizeData(rf)
                                ax[l//12, l%12].imshow(rf, cmap = 'gray')
                            plt.show()

                        self.W_HX = W_HX
                        self.W_YH = W_YH
                        self.M_H = M_H
                        self.M_Y = M_Y
                        self.D1 = D1
                        self.D2 = D2

                        self.H = H
                        self.Y = Y

                    except Exception as e:
                        print(e)

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y

    def fit_batch_general_polytope(self, X, signed_dims, nn_dims, sparse_dims_list, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, lambda_lr = 1, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []
        self.D2minlist = []
        self.D1minlist = []
        if self.Y is None:
            H = np.zeros((h_dim,samples)) 
            Y = np.zeros((s_dim,samples))
            # H = np.random.randn(h_dim, samples)
            # Y = np.random.randn(h_dim, samples)
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        if (signed_dims.size == 0):
            signed_dims = np.array([-1])
        if (nn_dims.size == 0):
            nn_dims = np.array([-1])
        if (not sparse_dims_list):
            sparse_dims_list = [np.array([-1])]

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y,_ = self.run_neural_dynamics_general_polytope_jit(x_current = x_current, h = h, y = y, 
                                                                      signed_dims = signed_dims, nn_dims = nn_dims, 
                                                                      sparse_dims_list = sparse_dims_list, M_H = M_H, M_Y = M_Y, 
                                                                      W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                      neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                                      lr_stop = neural_lr_stop, lambda_lr = lambda_lr, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                # D1 = D1 - muD[0] * D1derivative
                D1 = D1 - clipping(muD[0] * D1derivative, D1 * 1)

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                # D2 = D2 - muD[1] * D2derivative
                D2 = D2 - clipping(muD[1] * D2derivative, D2 * 1)

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            diagGG = np.diag(GG)
                            # Signal Power
                            sigpow = np.linalg.norm(diagGG,2)**2
                            # Interference Power
                            intpow = np.linalg.norm(GG, 'fro')**2 - sigpow

                            SIR = 10*np.log10(sigpow/intpow)

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            bias = + (S.T - Y_).mean(axis = 0)
                            Y_ = Y_ + bias
                            self.Y_ = Y_ 
                            
                            # SIR_list.append(SIR)
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S, False)[0]))

                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            self.D1minlist = D1minlist
                            self.D2minlist = D2minlist
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list     

class OnlineNSM:
    """
    Implementation of Online Nonnegative Similarity Matching.

    Parameters:
    ==========================================================
    s_dim           --Dimension of the sources
    x_dim           --Dimension of the mixtures
    W1
    W2
    Dt
    neural_OUTPUT_COMP_TOL
    set_ground_truth
    S               --Original Sources (for debugging only)
    A               --Mixing Matrix (for debugging only)

    Methods:
    ===========================================================
    whiten_input
    snr
    ZeroOneNormalizeData
    ZeroOneNormalizeColumns
    compute_overall_mapping
    CalculateSIR
    predict
    run_neural_dynamics
    fit_batch_nsm
    """
    def __init__(self, s_dim, x_dim, W1 = None, W2 = None, Dt = None, whiten_input_ = True, set_ground_truth = False, S = None, A = None):

        if W1 is not None:
            if whiten_input_:
                assert W1.shape == (s_dim, s_dim), "The shape of the initial guess W1 must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
                W1 = W1
            else:
                assert W1.shape == (s_dim, x_dim), "The shape of the initial guess W1 must be (s_dim, x_dim) = (%d, %d)" % (s_dim, x_dim)
                W1 = W1
        else:
            if whiten_input_:
                W1 = np.eye(s_dim, s_dim)
            else:
                W1 = np.eye(s_dim, x_dim)

        if W2 is not None:
            assert W2.shape == (s_dim, s_dim), "The shape of the initial guess W2 must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
            W2 = W2
        else:
            W2 = np.zeros((s_dim, s_dim))

        if Dt is not None:
            assert Dt.shape == (s_dim, 1), "The shape of the initial guess Dt must be (s_dim, 1) = (%d, %d)" % (s_dim, 1)
            Dt = Dt
        else:
            Dt = 0.1 * np.ones((s_dim, 1))

        
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W1 = W1
        self.W2 = W2
        self.Dt = Dt
        self.whiten_input_ = whiten_input_
        self.Wpre = np.eye(x_dim)
        self.set_ground_truth = set_ground_truth
        self.S = S
        self.A = A
        self.SIRlist = []

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
        return H


    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def ZeroOneNormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def ZeroOneNormalizeColumns(self,X):
        X_normalized = np.empty_like(X)
        for i in range(X.shape[1]):
            X_normalized[:,i] = self.ZeroOneNormalizeData(X[:,i])

        return X_normalized

    def compute_overall_mapping(self, return_mapping = False):
        W1, W2 = self.W1, self.W2
        Wpre = self.Wpre
        W = np.linalg.pinv(np.eye(self.s_dim) + W2) @ W1 @ Wpre
        self.W = W
        if return_mapping:
            return W

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

    def predict(self, X):
        Wf = self.compute_overall_mapping(return_mapping = True)
        return Wf @ X

    @staticmethod
    @njit
    def run_neural_dynamics(x, y, W1, W2, n_iterations = 200):
        for j in range(n_iterations):
            ind = math.floor((np.random.rand(1) * y.shape[0])[0])         
            y[ind, :] = np.maximum(np.dot(W1[ind, :], x) - np.dot(W2[ind, :], y), 0)

        return y

    def fit_batch_nsm(self, X, n_epochs = 1, neural_dynamic_iterations = 250, shuffle = True, debug_iteration_point = 100, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W1, W2, Dt = self.W1, self.W2, self.Dt
        debugging = self.set_ground_truth
        ZERO_CHECK_INTERVAL = 1500
        nzerocount = np.zeros(s_dim)
        whiten_input_ = self.whiten_input_

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        if whiten_input_:
            X_ = self.whiten_input(X)
            x_dim = X_.shape[0]
        else:
            X_ = X

        Wpre = self.Wpre

        if debugging:
            SIRlist = self.SIRlist
            S = self.S
            A = self.A

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)

            for i_sample in tqdm(range(samples)):
                x_current = X_[:, idx[i_sample]]
                xk = np.reshape(x_current, (-1,1))

                y = np.random.rand(s_dim, 1)

                y = self.run_neural_dynamics(xk, y, W1, W2, neural_dynamic_iterations)

                Dt = np.minimum(3000, 0.94 * Dt + y ** 2)
                DtD = np.diag(1 / Dt.reshape((s_dim)))
                W1 = W1 + np.dot(DtD, (np.dot(y, (xk.T).reshape((1, x_dim))) - np.dot(np.diag((y ** 2).reshape((s_dim))), W1)))
                W2 = W2 + np.dot(DtD, (np.dot(y, y.T) - np.dot(np.diag((y ** 2).reshape((s_dim))), W2)))

                for ind in range(s_dim):
                    W2[ind, ind] = 0

                nzerocount = (nzerocount + (y.reshape(s_dim) == 0) * 1.0) * (y.reshape(s_dim) == 0)
                if i_sample < ZERO_CHECK_INTERVAL:
                    q = np.argwhere(nzerocount > 50)
                    qq = q[:,0]
                    for iter3 in range(len(qq)):
                        W1[qq[iter3], :] = -W1[qq[iter3], :]
                        nzerocount[qq[iter3]] = 0

                self.W1 = W1
                self.W2 = W2
                self.Dt = Dt

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        # self.SIRlist = SIRlist

                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behavior", fontsize = 15)
                            pl.grid()
                            clear_output(wait = True)
                            display(pl.gcf())
 
class OnlineBSM:
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
    
    ProjectOntoLInfty(X)   -- Project the given vector X onto L_infinity norm ball
    
    fit_next_antisparse(x_online)     -- Updates the network parameters for one data point x_online
    
    fit_batch_antisparse(X_batch)     -- Updates the network parameters for given batch data X_batch (but in online manner)
    
    """
    def __init__(self, s_dim, x_dim, gamma = 0.9999, mu = 1e-3, beta = 1e-7, W = None, M = None, D = None, whiten_input_ = True, neural_OUTPUT_COMP_TOL = 1e-6, set_ground_truth = False, S = None, A = None):
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
                W = 0.0033 * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            else:
                W = np.random.randn(s_dim,x_dim)
                W = 0.0033 * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            # for k in range(W_HX.shape[0]):
            #     W_HX[k,:] = WScalings[0] * W_HX[k,:]/np.linalg.norm(W_HX[k,:])
            
        if M is not None:
            assert M.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M
        else:
            M = 0.02*np.eye(s_dim)  
            
        if D is not None:
            assert D.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            D = D
        else:
            D = 1*np.eye(s_dim)
            
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
        self.SIRlist = []
        
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
    
    def ProjectOntoLInfty(self, X):

        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def compute_overall_mapping(self,return_mapping = True):
        W, M, D = self.W, self.M, self.D

        Wf = np.linalg.pinv(M @ D) @ W @ self.Wpre
        self.Wf = np.real(Wf)

        if return_mapping:
            return np.real(Wf)
        else:
            return None

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):

        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            # u = u - lr * du
            y = y - lr * du

            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):

        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh-0*(X<0)
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoNNLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            # u = u - lr * du
            y = y - lr * du

            y = ProjectOntoNNLInfty(u / np.diag(Upsilon * D))

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    @staticmethod
    @njit
    def run_neural_dynamics_mixedantisparse(x, y, nn_components, signed_components, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):

        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh-0*(X<0)

        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            y = y - lr * du

            y = u / np.diag(Upsilon * D)
            y[nn_components] = ProjectOntoNNLInfty(y[nn_components])
            y[signed_components] = ProjectOntoLInfty(y[signed_components])

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    @staticmethod
    @njit
    def run_neural_dynamics_sparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):
        
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

        STLAMBD = 0
        dval = 0
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            # u = u - lr * du
            y = y - lr * du
            
            # SOFT THRESHOLDING
            y_absolute = np.abs(y)
            y_sign = np.sign(y)

            y = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign
            dval = np.linalg.norm(y, 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):

        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh-0*(X<0)
            
        STLAMBD = 0
        dval = 0
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoNNLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            # u = u - lr * du
            y = y - lr * du

            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = max(STLAMBD + 1.5 * dval, 0)

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubsparse(x, y, nn_components, sparse_components, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):
        
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=0)+(X>thresh)*thresh

        STLAMBD = 0
        dval = 0
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoNNLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            y = y - lr * du

            y[nn_components] = ProjectOntoNNLInfty(y[nn_components])
            # SOFT THRESHOLDING
            y_sparse_absolute = np.abs(y[sparse_components])
            y_sparse_sign = np.sign(y[sparse_components])

            y[sparse_components] = (y_sparse_absolute > STLAMBD) * (y_sparse_absolute - STLAMBD) * y_sparse_sign
            dval = np.linalg.norm(y[sparse_components], 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    def fit_next_antisparse(self, x_current, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False):
        W = self.W
        M = self.M
        D = self.D
        gamma, mu, beta = self.gamma, self.mu, self.beta
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        # Upsilon = np.diag(np.diag(M))
        
        # u = np.linalg.solve(M @ D, W @ x_current)
        # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
        y = np.random.randn(self.s_dim,)
        y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)

        
        W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
        M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
        
        D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
        
        self.W = W
        self.M = M
        self.D = D
        
    def fit_batch_antisparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
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
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

    def fit_batch_nnantisparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
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
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_nnantisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist


                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

    def fit_batch_mixedantisparse(self, X, nn_components, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
        else:
            S = None
            A = None

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]

        Y = 0.05*np.random.randn(self.s_dim, samples)
        
        source_indices = [j for j in range(self.s_dim)]
        signed_components = source_indices.copy()
        for a in nn_components:
            signed_components.remove(a)
        nn_components = np.array(nn_components)
        signed_components = np.array(signed_components)

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
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_mixedantisparse(x_current, y, nn_components, signed_components, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

    def fit_batch_sparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
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
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_sparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist


                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

    def fit_batch_nnsparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
        else:
            S = None
            A = None

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]

        Y = np.zeros((self.s_dim, samples)) + 0.5
        
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
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_nnsparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)

                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist


                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

    def fit_batch_nnwsubsparse(self, X, sparse_components, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A
        else:
            S = None
            A = None

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]

        Y = 0.05*np.random.randn(self.s_dim, samples)
        
        source_indices = [j for j in range(self.s_dim)]
        nn_components = source_indices.copy()
        for a in sparse_components:
            nn_components.remove(a)
        sparse_components = np.array(sparse_components)
        nn_components = np.array(nn_components)

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
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_nnwsubsparse(x_current, y, nn_components, sparse_components, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIRlist.append(10*np.log10(CalculateSINR(Y_.T, S)[0]))
                        self.SIRlist = SIRlist


                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

class OnlineWSMPMF(OnlineWSMBSS):
    @staticmethod
    @njit
    def run_neural_dynamics(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                           neural_dynamic_iterations, lr_start, lr_stop, OUTPUT_COMP_TOL):
        Gamma_H = np.diag(np.diag(M_H))
        M_hat_H = M_H - Gamma_H

        Gamma_Y = np.diag(np.diag(M_Y))
        M_hat_Y = M_Y - Gamma_Y

        v = ((1 - beta) * Gamma_H + beta * D1 @ Gamma_H @ D1) @ h
        u = Gamma_Y @ D2 @ y

        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        STLAMBD1 = 0
        STLAMBD2 = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)
            
            delv = -(1 - zeta) * v + (1 - zeta) * beta * D1 @ W_HX @ x_current
            delv = delv - ((1 - zeta) * (1 - beta) * M_hat_H + (1- zeta) * beta * D1 @ M_hat_H @ D1) @ h
            delv = delv + (1 - zeta) * (1 - beta) * W_YH.T @ D2 @ y
            v = v + MUV * delv
            h = v / np.diag(Gamma_H * ((1 - zeta) * (1 - beta) + (1 - zeta) * beta * D1 ** 2))
            h = h*(h>=-2.0)*(h<=2.0)+(h>2.0)*2.0-2.0*(h<-2.0)
            
            delu = -u + W_YH @ h
            delu = delu - M_hat_Y @ D2 @ y
            u = u + (MUV) * delu
            y = u / np.diag(Gamma_Y * (D2))
            
            y_sparse_absolute1 = np.abs(y[np.array([0,1])])
            y_sparse_absolute2 = np.abs(y[np.array([1,2])])
            y_sparse_sign1 = np.sign(y[np.array([0,1])])
            y_sparse_sign2 = np.sign(y[np.array([1,2])])
            
            y[0] = (y_sparse_absolute1[0] > STLAMBD1) * (y_sparse_absolute1[0] - STLAMBD1) * y_sparse_sign1[0]
            y[1] = (y_sparse_absolute1[1] > (STLAMBD1 + STLAMBD2)) * (y_sparse_absolute1[1] - (STLAMBD1 + STLAMBD2)) * y_sparse_sign1[1]
            y[2] = (y_sparse_absolute2[1] > STLAMBD2) * (y_sparse_absolute2[1] - STLAMBD2) * y_sparse_sign2[1]
            
            y = y*(y>=-1.0)*(y<=1.0)+(y>1.0)*1.0-1.0*(y<-1.0)
            y[2] = y[2]*(y[2]>=0)*(y[2]<=1) + 1.0*(y[2]>1)
            
            dval1 = np.linalg.norm(y[np.array([0,1])],1) - 1
            dval2 = np.linalg.norm(y[np.array([1,2])],1) - 1
            
            STLAMBD1 = max(STLAMBD1 + 1.5 * dval1,0)
            STLAMBD2 = max(STLAMBD2 + 1.5 * dval2,0)
            
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u
            
        return h,y, OutputCounter

    def fit_batch_pmf(self, X, n_epochs = 1, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.5, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma_start, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.beta, self.zeta, self.muD, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"

        s_dim = self.s_dim
        h_dim = self.h_dim
        samples = X.shape[1]
        D1minlist = []
        D2minlist = []
        self.SV_list = []
        self.D2minlist = []
        self.D1minlist = []
        if self.Y is None:
            H = np.zeros((h_dim,samples)) 
            Y = np.zeros((s_dim,samples))
            # H = np.random.randn(h_dim, samples)
            # Y = np.random.randn(h_dim, samples)
        else:
            H, Y = self.H, self.Y

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 
            plt.figure(figsize = (70, 50), dpi = 80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):

                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]

                h,y,_ = self.run_neural_dynamics(x_current = x_current, h = h, y = y, M_H = M_H, M_Y = M_Y, 
                                                          W_HX = W_HX, W_YH = W_YH, D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                          neural_dynamic_iterations = neural_dynamic_iterations, lr_start = neural_lr_start, 
                                                          lr_stop = neural_lr_stop, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                MUS = np.max([gamma_start/(1 + np.log(2 + i_sample)), 0.001])

                M_H = (1 - MUS) * M_H + MUS * np.outer(h,h)
                M_Y = (1 - MUS) * M_Y + MUS * np.outer(y,y)
                
                W_HX = (1 - MUS) * W_HX + MUS * np.outer(h,x_current)
                W_YH = (1 - MUS) * W_YH + MUS * np.outer(y,h)
                
                D1derivative = (1 - zeta) * beta * np.diag(np.diag(M_H @ D1 @ M_H - W_HX @ W_HX.T)) + zeta * self.dlogdet(D1)
                # D1 = D1 - muD[0] * D1derivative
                D1 = D1 - clipping(muD[0] * D1derivative, D1 * 1)

                D2derivative = (1 - zeta) * (1 - beta) * np.diag(np.diag(M_Y @ D2 @ M_Y - W_YH @ W_YH.T)) + zeta * self.dlogdet(D2)
                # D2 = D2 - muD[1] * D2derivative
                D2 = D2 - clipping(muD[1] * D2derivative, D2 * 1)

                d1 = np.diag(D1)
                d2 = np.diag(D2)
                D1 = np.diag(d1 * (d1 > LayerMinimumGains[0]) * (d1 < LayerMaximumGains[0]) + LayerMaximumGains[0] * (d1 >= LayerMaximumGains[0]) + LayerMinimumGains[0] * (d1 <= LayerMinimumGains[0]))
                D2 = np.diag(d2 * (d2 > LayerMinimumGains[1]) * (d2 < LayerMaximumGains[1]) + LayerMaximumGains[1] * (d2 >= LayerMaximumGains[1]) + LayerMinimumGains[1] * (d2 <= LayerMinimumGains[1]))
                
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h               

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping(return_mapping = True)

                            self.W = W

                            T = W @ A
                            Tabs = np.abs(T)
                            P = np.zeros((s_dim, s_dim))

                            for SourceIndex in range(s_dim):
                                Tmax = np.max(Tabs[SourceIndex,:])
                                Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
                                P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
                            
                            GG = P.T @ T
                            _, SGG, _ =np.linalg.svd(GG)
                            self.SV_list.append(abs(SGG))

                            diagGG = np.diag(GG)
                            # Signal Power
                            sigpow = np.linalg.norm(diagGG,2)**2
                            # Interference Power
                            intpow = np.linalg.norm(GG, 'fro')**2 - sigpow

                            SIR = 10*np.log10(sigpow/intpow)

                            Y_ = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            bias = + (S.T - Y_).mean(axis = 0)
                            Y_ = Y_ + bias
                            self.Y_ = Y_ 
                            
                            # SIR_list.append(SIR)
                            SIR_list.append(10*np.log10(CalculateSINR(Y_.T, S, False)[0]))

                            SNR_list.append(self.snr(S.T,Y_))
                            if plot_in_jupyter:
                                # d1_min, d1_max, d2_min, d2_max = np.min(np.diag(D1)), np.max(np.diag(D1)), np.min(np.diag(D2)), np.max(np.diag(D2))
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)
                                # D1maxlist.append(d1_max)
                                # D2maxlist.append(d2_max)

                                pl.clf()
                                pl.subplot(3,2,1)
                                pl.plot(np.array(SIR_list), linewidth = 5)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.ylabel("SIR (dB)", fontsize = 45)
                                pl.title("SIR Behaviour", fontsize = 45)
                                pl.grid()
                                # pl.title("Neural Dynamic Iteration Number : {}".format(str(oc)), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,3)
                                pl.plot(np.array(D1minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D1", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,4)
                                pl.plot(np.array(D2minlist), linewidth = 5)
                                pl.grid()
                                pl.title("Diagonal Values of D2", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,2)
                                pl.plot(np.array(SNR_list), linewidth = 5)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 45)
                                pl.ylabel("SNR (dB)", fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,5)
                                pl.plot(np.array(self.SV_list), linewidth = 5)
                                pl.grid()
                                pl.title("Singular Value Check, Overall Matrix Rank: "+str(np.linalg.matrix_rank(P)) , fontsize = 45)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                pl.subplot(3,2,6)
                                pl.plot(Y[:,idx[i_sample-25:i_sample]].T, linewidth = 5)
                                pl.title("Y last 25", fontsize = 45)
                                pl.grid()
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)

                                clear_output(wait=True)
                                display(pl.gcf())   

                            self.W_HX = W_HX
                            self.W_YH = W_YH
                            self.M_H = M_H
                            self.M_Y = M_Y
                            self.D1 = D1
                            self.D2 = D2

                            self.H = H
                            self.Y = Y
                            self.SIR_list = SIR_list
                            self.SNR_list = SNR_list 
                            self.D1minlist = D1minlist
                            self.D2minlist = D2minlist
                        except Exception as e:
                            print(str(e))
                            

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2

        self.H = H
        self.Y = Y
        self.SIR_list = SIR_list
        self.SNR_list = SNR_list               
     
def fit_icainfomax(X, NumberofSources = None, ch_types = None, n_subgauss = None, max_iter = 10000, verbose = False):
    """
    X : Mixture Signals, X.shape = (NumberofMixtures, NumberofSamples)
    
    for more information, visit:
    https://mne.tools/stable/generated/mne.preprocessing.ICA.html

    USAGE:
    Y = fit_icainfomax(X = X, NumberofSources = 3)
    IF GROUND TRUTH IS AVAILABLE:
    Y_ = signed_and_permutation_corrected_sources(S.T, Y.T).T
    """
    NumberofMixtures = X.shape[0]
    if NumberofSources is None:
        NumberofSources = NumberofMixtures
    if ch_types is None:
        ch_types = ["eeg"] * NumberofMixtures
    if n_subgauss is None:
        n_subgauss = NumberofSources
    mneinfo = mne.create_info(NumberofMixtures, 2000, ch_types = ch_types)
    mneobj = mne.io.RawArray(X, mneinfo)
    ica = mne.preprocessing.ICA(n_components = NumberofSources, method = "infomax",
                                fit_params = {"extended": True, "n_subgauss":n_subgauss,"max_iter":max_iter},
                                random_state = 1,verbose = verbose)
    ica.fit(mneobj)
    Se = ica.get_sources(mneobj)
    Y = Se.get_data()
    return Y

############################ REQUIRED FUNCTIONS ######################################

def whiten_signal(X, mean_normalize = True, type_ = 3):
    """
    Input : X  ---> Input signal to be whitened
    
    type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
    
    Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
    """
    if mean_normalize:
        X = X - np.mean(X,axis = 0, keepdims = True)
    
    cov = np.cov(X.T)
    
    if type_ == 3: # Whitening using singular value decomposition
        U,S,V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        W_pre = np.dot(U, np.dot(d, U.T))
        
    else: # Whitening using eigenvalue decomposition
        d,S = np.linalg.eigh(cov)
        D = np.diag(d)

        D_sqrt = np.sqrt(D * (D>0))

        if type_ == 1: # Type defines how you want W_pre matrix to be
            W_pre = np.linalg.pinv(S@D_sqrt)
        elif type_ == 2:
            W_pre = np.linalg.pinv(S@D_sqrt@S.T)
    
    X_white = (W_pre @ X.T).T
    
    return X_white, W_pre

def whiten_input(X, n_components = None, return_prewhitening_matrix = False):
    """
    X.shape[0] = Number of sources
    X.shape[1] = Number of samples for each signal
    """
    x_dim = X.shape[0]
    if n_components is None:
        n_components = x_dim
    s_dim = n_components
    
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
    if return_prewhitening_matrix:
        return H, Wpre
    else:
        return H

def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:,i] = ZeroOneNormalizeData(X[:,i])

    return X_normalized

def ProjectOntoLInfty(X):
    return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)

def Subplot_gray_images(I, image_shape = [512,512], height = 15, width = 15, title = ''):
    n_images = I.shape[1]
    fig, ax = plt.subplots(1,n_images)
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:,i].reshape(image_shape[0],image_shape[1]), cmap = 'gray')
    
    plt.show()

def subplot_1D_signals(X, title = '',title_fontsize = 20, figsize = (10,5), linewidth = 1, colorcode = '#050C12'):
    """
    Plot the 1D signals (each column from the given matrix)
    """
    n = X.shape[1] # Number of signals
    
    fig, ax = plt.subplots(n,1, figsize = figsize)
    
    for i in range(n):
        ax[i].plot(X[:,i], linewidth = linewidth, color = colorcode)
        ax[i].grid()
    
    plt.suptitle(title, fontsize = title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()

def plot_convergence_plot(metric, xlabel = '', ylabel = '', title = '', figsize = (12,8), fontsize = 15, linewidth = 3, colorcode = '#050C12'):
    
    plt.figure(figsize = figsize)
    plt.plot(metric, linewidth = linewidth, color = colorcode)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.title(title, fontsize = fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.draw()
    
def outer_prod_broadcasting(A, B):
    """Broadcasting trick"""
    return A[...,None]*B[:,None]

def find_permutation_between_source_and_estimation(S,Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
    
    return the permutation of the source seperation algorithm
    """
    
    # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    # perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    perm = np.argmax(np.abs(outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
    return perm

def signed_and_permutation_corrected_sources(S,Y):
    perm = find_permutation_between_source_and_estimation(S,Y)
    return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

def psnr(img1, img2, pixel_max = 1):
    """
    Return peak-signal-to-noise-ratio between given two images
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(pixel_max / np.sqrt(mse))

def snr(S_original, S_noisy):
    N_hat = S_original - S_noisy
    N_P = (N_hat ** 2).sum(axis = 0)
    S_P = (S_original ** 2).sum(axis = 0)
    snr = 10 * np.log10(S_P / N_P)
    return snr

def ProjectRowstoL1NormBall(H):
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

def display_matrix(array):
    data = ''
    for line in array:
        if len(line) == 1:
            data += ' %.3f &' % line + r' \\\n'
            continue
        for element in line:
            data += ' %.3f &' % element
        data += r' \\' + '\n'
    display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))

# Calculate SIR Function
def CalculateSIR(H,pH, return_db = True):
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

def CalculateSINR(Out,S, compute_permutation = True):
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
    
def addWGN(signal, SNR, return_noise = False, print_resulting_SNR = False):
    """
    Adding white Gaussian Noise to the input signal
    signal              : Input signal, numpy array of shape (number of sources, number of samples)
                          If your signal is a 1D numpy array of shape (number of samples, ), then reshape it 
                          by signal.reshape(1,-1) before giving it as input to this function
    SNR                 : Desired input signal to noise ratio
    print_resulting_SNR : If you want to print the numerically calculated SNR, pass it as True
    
    Returns
    ============================
    signal_noisy        : Output signal which is the sum of input signal and additive noise
    noise               : Returns the added noise
    """
    sigpow = np.mean(signal**2, axis = 1)
    noisepow = 10 **(-SNR/10) * sigpow
    noise =  np.sqrt(noisepow)[:,np.newaxis] * np.random.randn(signal.shape[0], signal.shape[1])
    signal_noisy = signal + noise
    if print_resulting_SNR:
        SNRinp = 10 * np.log10(np.sum(np.mean(signal**2, axis = 1)) / np.sum(np.mean(noise**2, axis = 1)))
        print("Input SNR is : {}".format(SNRinp))
    if return_noise:
        return signal_noisy, noise
    else:
        return signal_noisy

def WSM_Mixing_Scenario(S, NumberofMixtures = None, INPUT_STD = None):
    NumberofSources = S.shape[0]
    if INPUT_STD is None:
        INPUT_STD = S.std()
    if NumberofMixtures is None:
        NumberofMixtures = NumberofSources
    A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
    X = A @ S
    for M in range(A.shape[0]):
        stdx = np.std(X[M,:])
        A[M,:] = A[M,:]/stdx * INPUT_STD

    X = A @ S

    return A, X

def generate_correlated_uniform_sources(R, range_ = [-1,1], n_sources = 5, size_sources = 500000):
    """
    R : correlation matrix
    """
    assert R.shape[0] == n_sources, "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)" % (n_sources)
    S = np.random.uniform(range_[0], range_[1], size = (n_sources, size_sources))
    L = np.linalg.cholesky(R)
    S_ = L @ S
    return S_

def generate_correlated_copula_sources(rho = 0.0, df = 4, n_sources = 5, size_sources = 500000, decreasing_correlation = True):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho ** j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size = size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df/s) * Z # chi-square method
    S = t.cdf(X, df).T
    return S

def generate_uniform_points_in_polytope(polytope_vertices, size):
    """"
    polytope_vertices : vertex matrix of shape (n_dim, n_vertices)

    return:
        Samples of shape (n_dim, size)
    """
    polytope_vertices = polytope_vertices.T
    dims = polytope_vertices.shape[-1]
    hull = polytope_vertices[ConvexHull(polytope_vertices).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = size, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = size)).T

def generate_practical_polytope(dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list):
    A = []
    b = []
    for j in antisparse_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(1)

    for j in nonnegative_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(0)

    for relative_sparse_dims in relative_sparse_dims_list:
        row = np.zeros(dim)
        pm_one = [[1,-1] for _ in range(relative_sparse_dims.shape[0])]
        for i in itertools.product(*pm_one):
            row_copy = row.copy()
            row_copy[relative_sparse_dims] = i
            A.append(list(row_copy))
            b.append(1)
    A = np.array(A)
    b = np.array(b)
    vertices = pypoman.compute_polytope_vertices(A, b)
    V = np.array([list(v) for v in vertices]).T
    return (A,b), V
      
