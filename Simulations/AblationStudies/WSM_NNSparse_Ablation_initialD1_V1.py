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

pickle_name_for_results = "simulation_results_nnsparse_ablation_initalD1_V1.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
h_dim = s_dim
x_dim = NumberofMixtures
samples = N

SNRlevel = 30

NumAverages = 50

seed_list = np.array([777*i for i in range(1, NumAverages+1)])

D1start_list = np.array([4,8,12,16])
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
RESULTS_DF = pd.DataFrame(columns = ['trial', 'seed', 'Model', 'D1', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 1000 # SIR measurement per 10000 iteration

for iter1 in range(NumAverages):
    seed_ = seed_list[iter1]
    np.random.seed(seed_)
    trial = iter1
    S = 2*np.random.uniform(size = (NumberofSources,int(N)))-1
    S = ProjectRowstoL1NormBall(S.T).T
    S = S*(S>=0)
    INPUT_STD = 0.5

    A, Xn = WSM_Mixing_Scenario(S, NumberofMixtures, INPUT_STD)
    Noisecomp=np.random.randn(A.shape[0],S.shape[1])*np.power(10,-SNRlevel/20)*INPUT_STD
    X=Xn+Noisecomp
    SNRinp = 20*np.log10(np.std(Xn)/np.std(Noisecomp))

    for D1start_ in D1start_list:
        #######################################################
        #                   WSM                               #
        #######################################################

        try: # Try Except for SVD did not converge error
            MUS = 0.25

            gammaM_start = [MUS, MUS]
            gammaM_stop = [1e-3, 1e-3]
            gammaW_start = [MUS, MUS]
            gammaW_stop = [1e-3, 1e-3]

            OUTPUT_COMP_TOL = 1e-5
            MAX_OUT_ITERATIONS = 3000
            LayerGains = [D1start_, 1]
            LayerMinimumGains = [1e-6, 1]
            LayerMaximumGains = [1e6, 1.001]
            WScalings = [0.0033, 0.0033]
            GamScalings = [0.02, 0.02]
            zeta = 1e-4
            beta = 0.5
            muD = [20, 1e-2]

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
                                    DScalings=LayerGains,
                                    LayerMinimumGains=LayerMinimumGains,
                                    LayerMaximumGains=LayerMaximumGains,
                                    neural_OUTPUT_COMP_TOL=OUTPUT_COMP_TOL,
                                    set_ground_truth=True,
                                    S=S,
                                    A=A,
                                )
            with Timer() as t:
                modelWSM.fit_batch_nnsparse(
                                            X,
                                            n_epochs=1,
                                            neural_lr_start=0.5,
                                            neural_lr_stop=0.2,
                                            neural_fast_start=True,
                                            synaptic_lr_decay_divider=1,
                                            debug_iteration_point=debug_iteration_point,
                                            plot_in_jupyter=False,
                                        )

            ######### Evaluate the Performance of Online WSM Framework ###########################
            SINRlistWSM = modelWSM.SIR_list
            WfWSM = modelWSM.compute_overall_mapping(return_mapping = True)
            YWSM = WfWSM @ X
            SINRWSM, SNRWSM, _, _, _ = evaluate_bssV2(WfWSM, YWSM, A, S, mean_normalize_estimations = False)
            # ['trial', 'seed', 'Model', 'zeta', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time']
            WSM_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'WSM', 'D1' : D1start_,
                        'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : SNRinp, 
                        'execution_time' : t.interval}

        except Exception as e:
            print(str(e))
            WSM_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'WSM', 'D1' : D1start_,
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)

        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))