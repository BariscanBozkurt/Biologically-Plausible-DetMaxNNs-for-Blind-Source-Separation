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
import logging
from time import time
import os
from IPython.display import display, Latex, Math, clear_output
import pylab as pl
import pandas as pd
np.random.seed(345)
import warnings
warnings.filterwarnings("ignore")

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

    def dlogdet(self, D, DEPS = 5e-8):
        d = np.diag(np.diag(D + DEPS * np.eye(len(D))) ** (-1))
        return d

    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ###########################################
    ############################################################################################
    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

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

    def fit_batch_antisparse(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
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
                            SIR_list.append(10*np.log10(self.CalculateSINR(Y_.T, S)[0]))
                            if plot_in_jupyter:
                                d1_min, d2_min = np.diag(D1), np.diag(D2)
                                D1minlist.append(d1_min)
                                D2minlist.append(d2_min)

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

@njit 
def map_estimates_to_symbols(Y, symbols):
    Ysymbols = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            idx = np.argmin(np.abs(Y[i,j] - symbols))
            Ysymbols[i,j] = symbols[idx]
    return Ysymbols

def SER(S, Y):
    """
    Symbol Error Rate
    """
    return np.sum((S - Y) != 0) / (S.size)

NumAverages = 10
SNRlevel = 30 # Input SNR in dB
seed_list = np.array([100879*i for i in range(1, NumAverages+1)])

########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
WSM_SIR_DF = pd.DataFrame(columns = ['trial', 'seed', 'NMixtures', 'Model','SIR', 'SINR', 'SIRlist', 'SNR', 'S', 'A', 'Wf', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 1000 # SIR measurement per 10000 iteration
NumberofMixtures_ = 10
for iter1 in range(NumAverages):
    trial = iter1
    seed_ = seed_list[iter1]
    np.random.seed(seed_)
    S = 2 * (np.random.randint(0,4,(5, 400000))) - 3
    NumberofMixtures = NumberofMixtures_
    NumberofSources = S.shape[0]
    A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
    X = A @ S

    X, NoisePart = addWGN(X, SNRlevel, return_noise = True)

    SNRinp = 10 * np.log10(np.sum(np.mean((X - NoisePart)**2, axis = 1)) / np.sum(np.mean(NoisePart**2, axis = 1)))
    # print("Input SNR is : {}".format(SNRinp))
    try:
        gamma_start = 0.1
        gamma_stop = 1e-3
        OUTPUT_COMP_TOL = 1e-6
        MAX_OUT_ITERATIONS= 3000
        LayerGains = [0.5,0.5]
        LayerMinimumGains = [0.2,0.2]
        LayerMaximumGains = [1e6,25]
        WScalings = [0.005,0.005]
        GamScalings = [2,1]
        zeta = 5*1e-3
        beta = 0.5
        muD = [0.01, 0.01]

        s_dim = S.shape[0]
        x_dim = X.shape[0]
        h_dim = x_dim
        samples = S.shape[1]

        debug_iteration_point = 40000
        modelWSM = OnlineWSMBSS(s_dim = s_dim, x_dim = x_dim, h_dim = h_dim, 
                                gamma_start = gamma_start, gamma_stop = gamma_stop, beta = beta, zeta = zeta, 
                                muD = muD,WScalings = WScalings, GamScalings = GamScalings,
                                DScalings = LayerGains, LayerMinimumGains = LayerMinimumGains,
                                LayerMaximumGains = LayerMaximumGains,neural_OUTPUT_COMP_TOL = OUTPUT_COMP_TOL,
                                set_ground_truth = True, S = S, A = A)
        twsm0 = time()

        modelWSM.fit_batch_antisparse(X/3, n_epochs = 1, 
                                    neural_lr_start = 0.3,
                                    neural_lr_stop = 0.05,
                                    shuffle = False,
                                    debug_iteration_point = debug_iteration_point,
                                    plot_in_jupyter = False,
                                    )
        twsm1 = time()

        execution_time = (twsm1 - twsm0) / 60.0

        Wwsm = modelWSM.compute_overall_mapping(return_mapping = True)
        Y_ = Wwsm @ X
        Y_ = modelWSM.signed_and_permutation_corrected_sources(S.T,Y_.T)
        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
        Y_ = coef_ * Y_
        SNRwsm = modelWSM.snr(S.T, Y_)
        # SIRwsm = CalculateSIR(A, Wwsm)[0]
        SINRwsm = 10*np.log10(modelWSM.CalculateSINR(Y_.T, S)[0])
        Model = 'WSM'
        SIRlist = modelWSM.SIR_list
        WSM_dict = {'trial' : trial, 'seed' : seed_, 'NMixtures': NumberofMixtures, 'Model' : Model, 
                    'SIR' : None, 'SINR' : SINRwsm,
                    'SIRlist' : SIRlist, 'SNR' : SNRwsm,  'S' : S, 'A' :A, 'Wf': Wwsm, 'SNRinp' : SNRinp, 
                    'execution_time': (twsm1 - twsm0)/60.0}
    except Exception as e:
        print(str(e))
        WSM_dict = {'trial' : trial, 'seed' : seed_, 'NMixtures': NumberofMixtures
                    , 'Model' : 'WSM', 'SIR' : str(e), 'SINR' : -999,
                    'SIRlist' : None,  'SNR' : None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp, 
                    'execution_time': str(e)}

    WSM_SIR_DF = WSM_SIR_DF.append(WSM_dict, ignore_index = True)
    WSM_SIR_DF.to_pickle("./simulation_results_4PAM_averageSINRv3.pkl")
WSM_SIR_DF.to_pickle("./simulation_results_4PAM_averageSINRv3.pkl")