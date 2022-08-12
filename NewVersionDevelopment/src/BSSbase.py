import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from numba import njit

class BSSBaseClass:

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

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 1)
        S_P = (S_original ** 2).sum(axis = 1)
        snr = 10 * np.log10(S_P / N_P)
        return snr
        
    @staticmethod
    @njit( parallel=True )
    def snr_jit(S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 1)
        S_P = (S_original ** 2).sum(axis = 1)
        snr = 10 * np.log10(S_P / N_P)
        return snr


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
        perm = np.argmax(np.abs(self.outer_prod_broadcasting(Y.T,S.T).sum(axis = 0))/(np.linalg.norm(S,axis = 1)*np.linalg.norm(Y,axis=1)), axis = 0)
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
