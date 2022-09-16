import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from tqdm import tqdm
from scipy.stats import invgamma, chi2, t
from WSMBSSv2 import *
from numba import njit
from IPython import display
import pylab as pl
from scipy.signal import lfilter
import mne 
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings("ignore")
# np.random.seed(37896)
# %load_ext autoreload
# %autoreload 2
notebook_name = 'NNAntisparse_Copula3'

N = 500000
NumberofSources = 5
NumberofMixtures = 10

M = NumberofMixtures
r = NumberofSources
#Define number of sampling points
n_samples = N
#Degrees of freedom
df = 4

# Correlation values
rholist=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

SNR = 30 # dB
NoiseAmp = (10 ** (-SNR/20))# * np.sqrt(NumberofSources)

NumAverages = 100

seed_list = np.array([1004*i for i in range(25, NumAverages+26)])


########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
WSM_SIR_DF = pd.DataFrame(columns = ['rho','trial', 'seed', 'Model','SIR', 'SINR', 'SIRlist', 'SNR', 'S', 'A', 'Wf', 'SNRinp'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 10000 # SIR measurement per 10000 iteration

for iter1 in range(NumAverages):
    seed_ = seed_list[iter1]
    np.random.seed(seed_)
    iter0=-1
    trial = iter1
    for rho in (rholist):
        
        iter0=iter0+1
        
        S = generate_correlated_copula_sources(rho = rho, df = 4, n_sources = NumberofSources, size_sources = N , decreasing_correlation = False)
        INPUT_STD = 0.28
        A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
        X = A @ S
        for MM in range(A.shape[0]):
            stdx = np.std(X[MM,:])
            A[MM,:] = A[MM,:]/stdx * INPUT_STD
        Xn = A @ S
        Noisecomp=np.random.randn(A.shape[0],S.shape[1])*np.power(10,-SNR/20)*INPUT_STD
        X=Xn+Noisecomp
        SNRinp = 20*np.log10(np.std(Xn)/np.std(Noisecomp))
        #######################################################
        #                   WSM                               #
        #######################################################
        try: # Try Except for SVD did not converge error
            if rho > 0.4:
                MUS = 0.05
                gamma_stop = 5*1e-4
            else:
                MUS = 0.1
                gamma_stop = 1e-3
            OUTPUT_COMP_TOL = 1e-7
            MAX_OUT_ITERATIONS= 3000
            LayerGains = [1,1]
            LayerMinimumGains = [1e-3,1e-3]
            LayerMaximumGains = [1e6,20]
            WScalings = [0.0033,0.0033]
            GamScalings = [2,1]
            zeta = 1*1e-5
            beta = 0.5
            muD = [1, 1e-2]

            s_dim = S.shape[0]
            x_dim = X.shape[0]
            h_dim = s_dim
            samples = S.shape[1]
            W_HX = np.eye(h_dim, x_dim)
            W_YH = np.eye(s_dim, h_dim)

            modelWSM = OnlineWSMBSS(s_dim = s_dim, x_dim = x_dim, h_dim = h_dim, 
                     gamma_start = MUS, gamma_stop = gamma_stop, beta = beta, zeta = zeta, 
                     muD = muD,WScalings = WScalings, GamScalings = GamScalings,
                     W_HX = W_HX, W_YH = W_YH,
                     DScalings = LayerGains, LayerMinimumGains = LayerMinimumGains,
                     LayerMaximumGains = LayerMaximumGains,neural_OUTPUT_COMP_TOL = OUTPUT_COMP_TOL,
                     set_ground_truth = True, S = S, A = A)
            
            modelWSM.fit_batch_nnantisparse(X, n_epochs = 1, 
                             neural_dynamic_iterations = 500,
                             neural_lr_start = 0.75,
                             neural_lr_stop = 0.05,
                             neural_fast_start = False,
                             whiten = False,
                             debug_iteration_point = debug_iteration_point,
                             plot_in_jupyter = False,
                            )

            Wwsm = modelWSM.compute_overall_mapping(return_mapping = True)
            Y_ = Wwsm @ X
            Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
            Y_ = coef_ * Y_
            SNRwsm = modelWSM.snr(S.T, Y_)
            SIRwsm = CalculateSIR(A, Wwsm)[0]
            SINRwsm = 10*np.log10(CalculateSINR(Y_.T, S)[0])
            trial = iter1
            Model = 'WSM'
            # SIR = SIRldmi
            SIRlist = modelWSM.SIR_list
            WSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : Model, 'SIR' : SIRwsm, 'SINR' : SINRwsm,
                        'SIRlist' : SIRlist, 'SNR' : SNRwsm,  'S' : S, 'A' :A, 'Wf': Wwsm, 'SNRinp' : SNRinp}
        except Exception as e:
            print(str(e))
            WSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'WSM', 'SIR' : str(e), 'SINR' : -999,
                          'SIRlist' : None,  'SNR' : None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}

        #######################################################
        #                   NSM                               #
        #######################################################
        try: # Try Except for SVD did not converge error
            modelNSM = OnlineNSM(s_dim = s_dim, x_dim = x_dim, set_ground_truth = True, S = S, A = A)
            
            modelNSM.fit_batch_nsm(X = X)
            Wnsm = modelNSM.compute_overall_mapping(return_mapping = True)
            Y_ = Wnsm @ X
            Y_ = signed_and_permutation_corrected_sources(S.T,Y_.T)
            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
            Y_ = coef_ * Y_
            SIRnsm = CalculateSIR(A, Wnsm)[0]
            SINRnsm = 10*np.log10(CalculateSINR(Y_.T, S)[0])
            SNRnsm = snr(S.T, Y_)
            SIRlist = modelNSM.SIRlist
            # 'rho','trial', 'seed', 'Model','SIR', 'SINR', 'SIRlist', 'SNR', 'S', 'A', 'Wf', 'SNRinp'
            NSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'NSM', 'SIR' : SIRnsm,
                        'SINR' : SINRnsm, 'SIRlist' : SIRlist, 'SNR': SNRnsm, 'S' : S, 'A' :A, 'Wf' : Wnsm, 'SNRinp' : SNRinp}
        except Exception as e:
            print(str(e))
            NSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'NSM', 'SIR' : str(e), 'SINR': -999,
                        'SIRlist' : None,'SNR': None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}

        #######################################################
        #                 ICA INFOMAX                         #
        #######################################################
        try:
            mneinfo=mne.create_info(M,2000,ch_types=["eeg"]*M)
            mneobj=mne.io.RawArray(X,mneinfo)
            ica = mne.preprocessing.ICA(n_components=r, method="infomax",
                                fit_params={"extended": True, "n_subgauss":r,"max_iter":10000},
                                random_state=1,verbose=True)
            ica.fit(mneobj)
            
            #SINR,SigPow,MSE,G=CalculateSINR(o,S)
            Se = ica.get_sources(mneobj)
            o = Se.get_data()
            SINR = 10*np.log10(CalculateSINR(o, S)[0])
            SIRICA = SINR
            SNRICA = snr(S.T, o.T)
            ICA_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'ICA', 'SIR' : SIRICA,
                        'SINR': SINR, 'SIRlist' : None, 'SNR': SNRICA, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}
        except Exception as e:
            print(str(e))
            ICA_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'ICA', 'SIR' : str(e),
                        'SINR': -999, 'SIRlist' : None, 'SNR': None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}


        WSM_SIR_DF = WSM_SIR_DF.append(WSM_dict, ignore_index = True)
        WSM_SIR_DF = WSM_SIR_DF.append(NSM_dict, ignore_index = True)
        WSM_SIR_DF = WSM_SIR_DF.append(ICA_dict, ignore_index = True)
        WSM_SIR_DF.to_pickle("./simulation_results_correlated_nnantisparseV9.pkl")

WSM_SIR_DF.to_pickle("./simulation_results_correlated_nnantisparseV9.pkl")