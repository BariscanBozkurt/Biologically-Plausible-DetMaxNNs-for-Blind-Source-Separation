########## IMPORT REQUIIRED LIBRARIES ##########
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from numba import njit
########## IMPORT UTILITY SCRIPTS ##############
import sys
sys.path.insert(0, '../../src')
from general_utils import *
from dsp_utils import *
from bss_utils import *
from numba_utils import *
######## IMPORT THE REQUIRED ALGORITHMS ########
from WSMBSS import OnlineWSMBSS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_4PAM.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)

NumAverages = 20 ## Number of realizations to average for each algorithm
seed_list = np.array([1574*i for i in range(1, NumAverages+1)]) ## Seeds for reproducibility
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1

    ###### GAUSSIAN MIXING SCENARIO AND WGN NOISE ADDING ##############################
    S = 2 * (np.random.randint(0,4,(NumberofSources, 400000))) - 3

    A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
    X = A @ S

    Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True) ## Add White Gaussian Noise with 30 dB SNR
    SNRinplevel = 10 * np.log10(np.sum(np.mean((Xnoisy - NoisePart) ** 2, axis = 1)) / np.sum(np.mean(NoisePart ** 2, axis = 1)))

    #######################################################
    #                   WSM                               #
    #######################################################
    try:
        gamma_start = 0.1
        gamma_stop = 1e-3
                
        gammaM_start = [gamma_start, gamma_start]
        gammaM_stop = [gamma_stop, gamma_stop]
        gammaW_start = [gamma_start, gamma_start]
        gammaW_stop = [gamma_stop, gamma_stop]

        OUTPUT_COMP_TOL = 1e-6
        MAX_OUT_ITERATIONS = 3000
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
        h_dim = s_dim
        samples = S.shape[1]

        modelWSM = OnlineWSMBSS(
                                s_dim=s_dim,
                                x_dim=x_dim,
                                h_dim=h_dim,
                                gammaM_start=gammaM_start,
                                gammaM_stop=gammaM_stop,
                                gammaW_start=gammaW_start,
                                gammaW_stop=gammaW_stop,
                                beta=beta,
                                zeta=zeta,
                                muD=muD,
                                WScalings=WScalings,
                                GamScalings=GamScalings,
                                DScalings=LayerGains,
                                LayerMinimumGains=LayerMinimumGains,
                                LayerMaximumGains=LayerMaximumGains,
                                neural_OUTPUT_COMP_TOL=OUTPUT_COMP_TOL,
                                set_ground_truth=True,
                                S=S,
                                A=A,
                            )

        with Timer() as t:
            modelWSM.fit_batch_antisparse(
                                            Xnoisy/3,
                                            n_epochs=1,
                                            neural_lr_start=0.3,
                                            neural_lr_stop=0.05,
                                            debug_iteration_point=debug_iteration_point,
                                            shuffle = False,
                                            plot_in_jupyter=False,
                                        )
        ######### Evaluate the Performance of Online WSM Framework ###########################
        SINRlistWSM = modelWSM.SIR_list
        WfWSM = modelWSM.compute_overall_mapping(return_mapping = True)
        YWSM = WfWSM @ Xnoisy
        SINRWSM, SNRWSM, _, _, _ = evaluate_bss(WfWSM, YWSM, A, S, mean_normalize_estimations = False)

        WSM_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                    'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                    'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : None, 
                    'execution_time' : t.interval}

    except Exception as e:
        print(str(e))
        WSM_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                    'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                    'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                    'execution_time' : None}

    RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)
    RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))