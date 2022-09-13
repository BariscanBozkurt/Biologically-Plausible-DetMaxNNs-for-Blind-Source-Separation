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
NumberofSources = 3
NumberofMixtures = 6

M = NumberofMixtures
r = NumberofSources
#Define number of sampling points
n_samples = N
#Degrees of freedom
df = 4

# Correlation values
rholist=np.array([0.0])

SNR = 30 # dB
NoiseAmp = (10 ** (-SNR/20))# * np.sqrt(NumberofSources)

NumAverages = 100

seed_list = np.array([14235794*i for i in range(25, NumAverages+26)])

dim = 3

antisparse_dims = np.array([0,1])
nonnegative_dims = np.array([2])
relative_sparse_dims_list = [np.array([0,1]),np.array([1,2])]
(A,b), V = generate_practical_polytope(dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list)
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
WSM_SIR_DF = pd.DataFrame(columns = ['rho','trial', 'seed', 'Model','SIR', 'SINR', 'SIRlist', 'SNR', 'SNRlist', 'S', 'A', 'Wf', 'SNRinp'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 10000 # SIR measurement per 10000 iteration

for iter1 in range(NumAverages):
    seed_ = seed_list[iter1]
    np.random.seed(seed_)
    iter0=-1
    trial = iter1
    for rho in (rholist):
        
        iter0=iter0+1
        
        S = generate_uniform_points_in_polytope(V, n_samples)

        INPUT_STD = np.std(S, axis = 1).mean()
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
            MUS = 0.25
            OUTPUT_COMP_TOL = 1e-5
            MAX_OUT_ITERATIONS= 3000
            LayerGains = [4,1]
            LayerMinimumGains = [1e-6,1]
            LayerMaximumGains = [1e6,1.001]
            WScalings = [0.0033,0.0033]
            GamScalings = [0.02,0.02]
            zeta = 1*1e-4
            beta = 0.5
            muD = [5.725, 1e-2]

            s_dim = S.shape[0]
            x_dim = X.shape[0]
            h_dim = s_dim
            samples = S.shape[1]
            debug_iteration_point = 10000
            modelWSM = OnlineWSMPMF(s_dim = s_dim, x_dim = x_dim, h_dim = h_dim, 
                                    gamma_start = MUS, beta = beta, zeta = zeta, 
                                    muD = muD,WScalings = WScalings, GamScalings = GamScalings,
                                    DScalings = LayerGains, LayerMinimumGains = LayerMinimumGains,
                                    LayerMaximumGains = LayerMaximumGains,neural_OUTPUT_COMP_TOL = OUTPUT_COMP_TOL,
                                    set_ground_truth = True, S = S, A = A)

            modelWSM.fit_batch_pmf(X, n_epochs = 1, 
                                    neural_lr_start = 0.3,
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
                        'SIRlist' : SIRlist, 'SNR' : SNRwsm, 'SNRlist' : modelWSM.SNR_list, 'S' : S, 'A' :A, 'Wf': Wwsm, 'SNRinp' : SNRinp}
        except Exception as e:
            print(str(e))
            WSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'WSM', 'SIR' : str(e), 'SINR' : -999,
                          'SIRlist' : None,  'SNR' : None, 'SNRlist' : None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}


        WSM_SIR_DF = WSM_SIR_DF.append(WSM_dict, ignore_index = True)
        WSM_SIR_DF.to_pickle("./simulation_results_specialpolyV5.pkl")

WSM_SIR_DF.to_pickle("./simulation_results_specialpolyV5.pkl")