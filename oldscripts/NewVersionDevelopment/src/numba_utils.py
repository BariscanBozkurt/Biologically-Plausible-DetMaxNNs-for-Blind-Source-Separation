###### NUMBA UTILITY FUNCTIONS FOR FASTER IMPLEMENTATION ###################
import numpy as np
from numba import njit

@njit
def CalculateSINRjit(Out,S, compute_permutation = True):
    """_summary_

    Args:
        Out (_type_): _description_
        S (_type_): _description_
        compute_permutation (bool, optional): _description_. Defaults to True.
    """
    def mean_numba(a):
        """_summary_

        Args:
            a (_type_): _description_

        Returns:
            _type_: _description_
        """
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
   
@njit( parallel=True )
def snr_jit(S_original, S_noisy):
    N_hat = S_original - S_noisy
    N_P = (N_hat ** 2).sum(axis = 1)
    S_P = (S_original ** 2).sum(axis = 1)
    snr = 10 * np.log10(S_P / N_P)
    return snr
  