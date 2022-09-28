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
from polytope_utils import *
######## IMPORT THE REQUIRED ALGORITHMS ########
from WSMBSS import OnlineWSMBSS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_correlated_antisparseV3.pkl"

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
(Apoly,bpoly), Vertices = generate_practical_polytope(dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list)
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
RESULTS_DF = pd.DataFrame(columns = ['rho','trial', 'seed', 'Model','SIR', 'SINR', 'SIRlist', 'SNR', 'SNRlist', 'S', 'A', 'Wf', 'SNRinp'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 10000 # SIR measurement per 10000 iteration

for iter1 in range(NumAverages):
    seed_ = seed_list[iter1]
    np.random.seed(seed_)
    iter0=-1
    trial = iter1
    for rho in (rholist):
        
        iter0=iter0+1
        
        S = generate_uniform_points_in_polytope(Vertices, n_samples)

        # INPUT_STD = np.std(S, axis = 1).mean()
        INPUT_STD = 0.5
        A, Xn = WSM_Mixing_Scenario(S, NumberofMixtures, INPUT_STD)
        Noisecomp=np.random.randn(A.shape[0],S.shape[1])*np.power(10,-SNR/20)*INPUT_STD
        X=Xn+Noisecomp
        SNRinp = 20*np.log10(np.std(Xn)/np.std(Noisecomp))
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
            LayerGains = [4, 1]
            LayerMinimumGains = [1e-6, 1]
            LayerMaximumGains = [1e6, 1.001]
            WScalings = [0.0033, 0.0033]
            GamScalings = [0.02, 0.02]
            zeta = 1 * 1e-4
            beta = 0.5
            muD = [5.725, 0]

            s_dim = S.shape[0]
            x_dim = X.shape[0]
            h_dim = s_dim
            samples = S.shape[1]
            # OPTIONS FOR synaptic_lr_rule: "constant", "divide_by_log_index", "divide_by_index"
            synaptic_lr_rule = "divide_by_log_index"
            # OPTIONS FOR neural_loop_lr_rule: "constant", "divide_by_loop_index", "divide_by_slow_loop_index"
            neural_loop_lr_rule = "constant"
            debug_iteration_point = 10000
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


            modelWSM.fit_batch_general_polytope(
                                                X,
                                                n_epochs=1,
                                                signed_dims=antisparse_dims,
                                                nn_dims=nonnegative_dims,
                                                sparse_dims_list=relative_sparse_dims_list,
                                                neural_lr_start=0.5,
                                                synaptic_lr_rule=synaptic_lr_rule,
                                                neural_loop_lr_rule=neural_loop_lr_rule,
                                                debug_iteration_point=debug_iteration_point,
                                                plot_in_jupyter=False,
                                            )

            Wf = modelWSM.compute_overall_mapping(return_mapping=True)
            Y = Wf @ X
            Y_ = signed_and_permutation_corrected_sources(S, Y)
            coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
            Y_ = coef_ * Y_

            SNRwsm = snr_jit(S, Y_)
            SIRwsm = CalculateSIR(A, Wf)[0]
            SINRwsm = 10 * np.log10(CalculateSINRjit(Y_, S)[0])
            trial = iter1
            Model = 'WSM'
            SIRlist = modelWSM.SIR_list
            WSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : Model, 'SIR' : SIRwsm, 'SINR' : SINRwsm,
                        'SIRlist' : SIRlist, 'SNR' : SNRwsm, 'SNRlist' : modelWSM.SNR_list, 
                        'S' : None, 'A' :None, 'Wf': Wf, 'SNRinp' : SNRinp}
        except Exception as e:
            print(str(e))
            WSM_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'WSM', 'SIR' : str(e), 'SINR' : -999,
                          'SIRlist' : None,  'SNR' : None, 'SNRlist' : None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}


        RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))