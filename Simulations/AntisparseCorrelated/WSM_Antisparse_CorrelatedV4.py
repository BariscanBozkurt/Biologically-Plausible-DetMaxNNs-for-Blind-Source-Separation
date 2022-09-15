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
from PMF import PMF, PMFv2
from LDMIBSS import LDMIBSS
from BSMBSS import OnlineBSM
from ICA import fit_icainfomax
from WSMBSS import OnlineWSMBSS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_correlated_antisparseV4.pkl"

N = 500000 ## Number of data points
NumberofSources = 4 ## Number of source vectors
NumberofMixtures = 8 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

rholist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) ## Correlation parameters
SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)

NumAverages = 50 ## Number of realizations to average for each algorithm
seed_list = np.array([444*i for i in range(1, NumAverages+1)]) ## Seeds for reproducibility
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['rho', 'trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1
    for rho in (rholist):
        ###### GAUSSIAN MIXING SCENARIO AND WGN NOISE ADDING ##############################
        S = generate_correlated_copula_sources(rho = rho, df = 4, n_sources = NumberofSources, 
                                               size_sources = N , decreasing_correlation = True) ## GENERATE CORRELATED COPULA
        S = 2 * S - 1
        A = np.random.randn(NumberofMixtures,NumberofSources)
        X = np.dot(A,S)
        Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True) ## Add White Gaussian Noise with 30 dB SNR
        SNRinplevel = 10 * np.log10(np.sum(np.mean((Xnoisy - NoisePart) ** 2, axis = 1)) / np.sum(np.mean(NoisePart ** 2, axis = 1)))

        #######################################################
        #                   WSM                               #
        #######################################################
        try:
            WSM_INPUT_STD = 0.5
            if rho > 0.4:
                gamma_start = 0.25
                gamma_stop = 5 * 1e-4
            else:
                gamma_start = 0.6
                gamma_stop = 1e-3

            gammaM_start = [gamma_start, gamma_start]
            gammaM_stop = [gamma_stop, gamma_stop]
            gammaW_start = [gamma_start, gamma_start]
            gammaW_stop = [gamma_stop, gamma_stop]

            OUTPUT_COMP_TOL = 1e-6
            MAX_OUT_ITERATIONS = 3000
            LayerGains = [1, 1]
            LayerMinimumGains = [0.2, 0.2]
            LayerMaximumGains = [1e6, 5]
            WScalings = [0.005, 0.005]
            GamScalings = [2, 1]
            zeta = 5 * 1e-5
            beta = 0.5
            muD = [1.125, 0.2]

            s_dim = S.shape[0]
            x_dim = X.shape[0]
            h_dim = s_dim
            samples = S.shape[1]
            W_HX = np.eye(h_dim, x_dim)
            W_YH = np.eye(s_dim, h_dim)

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
                                    W_HX=W_HX,
                                    W_YH=W_YH,
                                    DScalings=LayerGains,
                                    LayerMinimumGains=LayerMinimumGains,
                                    LayerMaximumGains=LayerMaximumGains,
                                    neural_OUTPUT_COMP_TOL=OUTPUT_COMP_TOL,
                                    set_ground_truth=True,
                                    S=S,
                                    A=A,
                                )

            XnoisyWSM = (WSM_INPUT_STD * (Xnoisy / Xnoisy.std(1)[:,np.newaxis]))
            with Timer() as t:
                modelWSM.fit_batch_antisparse(
                                                XnoisyWSM,
                                                n_epochs=1,
                                                neural_lr_start=0.75,
                                                neural_lr_stop=0.05,
                                                synaptic_lr_decay_divider=5,
                                                debug_iteration_point=debug_iteration_point,
                                                plot_in_jupyter=False,
                                            )

            ######### Evaluate the Performance of Online WSM Framework ###########################
            SINRlistWSM = modelWSM.SIR_list
            WfWSM = modelWSM.compute_overall_mapping(return_mapping = True)
            YWSM = WfWSM @ XnoisyWSM
            SINRWSM, SNRWSM, _, _, _ = evaluate_bss(WfWSM, YWSM, A, S, mean_normalize_estimations = False)

            WSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                        'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : None, 
                        'execution_time' : t.interval}

        except Exception as e:
            print(str(e))
            WSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                   BSM                               #
        #######################################################
        try: # Try Except for SVD did not converge error
            gamma = np.sqrt(1 - 4e-3)
            modelBSM = OnlineBSM(   s_dim = s_dim, x_dim = x_dim, beta = 1e-6, 
                                    gamma = gamma, whiten_input_ = True,
                                    W = np.eye(s_dim), M = np.eye(s_dim),
                                    neural_OUTPUT_COMP_TOL = 1e-7,
                                    set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelBSM.fit_batch_antisparse(  X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 10,
                                                neural_lr_start = 0.9, neural_lr_stop = 1e-15, 
                                                fast_start = True, debug_iteration_point = debug_iteration_point,
                                                plot_in_jupyter = False)

            ######### Evaluate the Performance of NSM Framework ###########################
            SINRlistBSM = modelBSM.SIR_list 
            WfBSM = modelBSM.compute_overall_mapping(return_mapping = True)
            YBSM = WfBSM @ Xnoisy
            SINRBSM, SNRBSM, _, _, _ = evaluate_bss(WfBSM, YBSM, A, S, mean_normalize_estimations = False)
            
            BSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'BSM',
                        'SINR' : SINRBSM, 'SINRlist':  SINRlistBSM, 'SNR' : SNRBSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfBSM, 'SNRinp' : None, 
                        'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            BSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'BSM',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                 ICA INFOMAX                         #
        #######################################################
        try:
            with Timer() as t:
                YICA = fit_icainfomax(Xnoisy, s_dim)

            ######### Evaluate the Performance of InfoMax-ICA Framework ###########################
            SINRlistICA = None 
            WfICA = YICA @ np.linalg.pinv(Xnoisy)
            SINRICA, SNRICA, _, _, _ = evaluate_bss(WfICA, YICA, A, S, mean_normalize_estimations = False)
            
            ICA_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'ICA',
                        'SINR' : SINRICA, 'SINRlist':  SINRlistICA, 'SNR' : SNRICA,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfICA, 'SNRinp' : None, 
                        'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            ICA_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'ICA',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                 LDMI BATCH                          #
        #######################################################
        try:
            modelLDMI = LDMIBSS(s_dim = s_dim, x_dim = x_dim,
                                set_ground_truth = True, S = S[:,:10000], A = A)
            with Timer() as t:
                ## Feed 10000 samples of the mixtures, that is enough for LDMI
                modelLDMI.fit_batch_antisparse(  Xnoisy[:,:10000], epsilon = 1e-5, mu_start = 100, n_iterations = 10000, 
                                                 method = "correlation", debug_iteration_point = debug_iteration_point,
                                                 plot_in_jupyter = False)
            
            ######### Evaluate the Performance of LDMIBSS Framework ###########################
            SINRlistLDMI = modelLDMI.SIR_list 
            WfLDMI = modelLDMI.W
            YLDMI = WfLDMI @ Xnoisy
            SINRLDMI, SNRLDMI, _, _, _ = evaluate_bss(WfLDMI, YLDMI, A, S, mean_normalize_estimations = False)
            
            LDMI_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : SINRLDMI, 'SINRlist':  SINRlistLDMI, 'SNR' : SNRLDMI,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfLDMI, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            LDMI_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                         'execution_time' : None}

        #######################################################
        #                 PMF BATCH                           #
        #######################################################
        try:
            modelPMF = PMFv2(s_dim = s_dim, y_dim = x_dim,
                             set_ground_truth = True, Sgt = S[:,:10000], Agt = A)
            with Timer() as t:
                modelPMF.fit_batch_antisparse(  Xnoisy[:,:10000], n_iterations = 100000,
                                                step_size_scale = 100,
                                                debug_iteration_point = debug_iteration_point,
                                                plot_in_jupyter = False)
            ######### Evaluate the Performance of PMF Framework ###########################
            SINRlistPMF = modelPMF.SIR_list 
            WfPMF = modelPMF.W
            # YPMF = modelPMF.S
            YPMF = WfPMF @ Xnoisy
            SINRPMF, SNRPMF, _, _, _ = evaluate_bss(WfPMF, YPMF, A, S, mean_normalize_estimations = False)
            
            PMF_Dict = { 'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                         'SINR' : SINRPMF, 'SINRlist':  SINRlistPMF, 'SNR' : SNRPMF,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfPMF, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            PMF_Dict = { 'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                         'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                         'execution_time' : None}

        RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(BSM_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(ICA_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(LDMI_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(PMF_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))