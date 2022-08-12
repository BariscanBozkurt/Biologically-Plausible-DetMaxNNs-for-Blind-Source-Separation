import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from numba import njit
from IPython.display import display, Latex, Math, clear_output
import pylab as pl
##### IMPORT MY UTILITY SCRIPTS #######
from BSSbase import *
from dsp_utils import *
from bss_utils import *
# from general_utils import *
from numba_utils import *
# from visualization_utils import * 

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15


@njit
def clipping(inp,lev):
    out=inp*(np.abs(inp)<=lev)+lev*(inp>lev)-lev*(inp<-lev)
    return out

################ WEIGHTED SIMILARITY MATCHING BLIND SOURCE SEPARATION #######################
class OnlineWSMBSS(BSSBaseClass):
    """_summary_
    """
    def __init__(self, s_dim, x_dim, h_dim = None, gammaW_start = [0.2, 0.2], gammaW_stop = [0.001,0.001], 
                 gammaM_start = [0.2, 0.2], gammaM_stop = [0.001,0.001], beta = 0.5, zeta = 1e-4, muD = [25,25], 
                 W_HX = None, W_YH = None, M_H = None, M_Y = None, D1 = None, D2 = None, WScalings = [0.0033,0.0033], 
                 GamScalings = [0.02, 0.02], DScalings = [25,1], LayerMinimumGains = [1e-6,1], LayerMaximumGains = [1e6,1], 
                 neural_OUTPUT_COMP_TOL = 1e-5, set_ground_truth = False, S = None, A = None ):
        """_summary_

        Args:
            s_dim (_type_): _description_
            x_dim (_type_): _description_
            h_dim (_type_, optional): _description_. Defaults to None.
            gamma_start (float, optional): _description_. Defaults to 0.2.
            gamma_stop (float, optional): _description_. Defaults to 0.001.
            beta (float, optional): _description_. Defaults to 0.5.
            zeta (_type_, optional): _description_. Defaults to 1e-4.
            muD (list, optional): _description_. Defaults to [25,25].
            W_HX (_type_, optional): _description_. Defaults to None.
            W_YH (_type_, optional): _description_. Defaults to None.
            M_H (_type_, optional): _description_. Defaults to None.
            M_Y (_type_, optional): _description_. Defaults to None.
            D1 (_type_, optional): _description_. Defaults to None.
            D2 (_type_, optional): _description_. Defaults to None.
            WScalings (list, optional): _description_. Defaults to [0.0033,0.0033].
            GamScalings (list, optional): _description_. Defaults to [0.02, 0.02].
            DScalings (list, optional): _description_. Defaults to [25,1].
            LayerMinimumGains (list, optional): _description_. Defaults to [1e-6,1].
            LayerMaximumGains (list, optional): _description_. Defaults to [1e6,1].
            neural_OUTPUT_COMP_TOL (_type_, optional): _description_. Defaults to 1e-5.
            set_ground_truth (bool, optional): _description_. Defaults to False.
            S (_type_, optional): _description_. Defaults to None.
            A (_type_, optional): _description_. Defaults to None.
        """
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
            assert D1.shape == (h_dim, 1), "The shape of the initial guess D must be (h_dim,h_dim)=(%d,%d)" % (h_dim, h_dim)
            D1 = D1
        else:
            D1 = DScalings[0] * np.ones((h_dim,1))

        if D2 is not None:
            assert D2.shape == (s_dim, 1), "The shape of the initial guess D must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            D2 = D2
        else:
            D2 = DScalings[1] * np.ones((s_dim,1))

        self.s_dim = s_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        if isinstance(gammaM_start,list):
            gammaM_start = np.array(gammaM_start)
        if isinstance(gammaM_stop,list):
            gammaM_stop = np.array(gammaM_stop)
        if isinstance(gammaW_start,list):
            gammaW_start = np.array(gammaW_start)
        if isinstance(gammaW_stop,list):
            gammaW_stop = np.array(gammaW_stop)
        self.gammaM_start = gammaM_start
        self.gammaM_stop = gammaM_stop
        self.gammaW_start = gammaW_start
        self.gammaW_stop = gammaW_stop
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
        if isinstance(LayerMinimumGains,list):
            LayerMinimumGains = np.array(LayerMinimumGains)
        if isinstance(LayerMaximumGains,list):
            LayerMaximumGains = np.array(LayerMaximumGains)
        self.LayerMinimumGains = LayerMinimumGains
        self.LayerMaximumGains = LayerMaximumGains
        self.Y = None
        self.H = None
        self.WScalings = WScalings
        self.sample_index = 0
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []
    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ###########################################
    ############################################################################################
    def evaluate_for_debug(self, W, A, S, X, mean_normalize_estimation = False):
        
        s_dim = self.s_dim
        Y_ = W @ X
        if mean_normalize_estimation:
            Y_ = Y_ - Y_.mean(axis = 1).reshape(-1,1)
        Y_ = self.signed_and_permutation_corrected_sources(S,Y_)
        coef_ = ((Y_ * S).sum(axis = 1) / (Y_ * Y_).sum(axis = 1)).reshape(-1,1)
        Y_ = coef_ * Y_

        SINR = 10*np.log10(self.CalculateSINRjit(Y_, S, False)[0])
        SNR = self.snr_jit(S, Y_)

        T = W @ A
        Tabs = np.abs(T)
        P = np.zeros((s_dim, s_dim))

        for SourceIndex in range(s_dim):
            Tmax = np.max(Tabs[SourceIndex,:])
            Tabs[SourceIndex,:] = Tabs[SourceIndex,:]/Tmax
            P[SourceIndex,:] = Tabs[SourceIndex,:]>0.999
        
        GG = P.T @ T
        _, SGG, _ = np.linalg.svd(GG) # SGG is the singular values of overall matrix Wf @ A

        return SINR, SNR, SGG, Y_, P

    def plot_for_debug(self, SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot):
        pl.clf()
        pl.subplot(3,2,1)
        pl.plot(np.array(SIR_list), linewidth = 5)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.ylabel("SIR (dB)", fontsize = 45)
        pl.title("SIR Behaviour", fontsize = 45)
        pl.grid()
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


        pl.subplot(3,2,3)
        pl.plot(np.array(D1list), linewidth = 5)
        pl.grid()
        pl.title("Diagonal Values of D1", fontsize = 45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(3,2,4)
        pl.plot(np.array(D2list), linewidth = 5)
        pl.grid()
        pl.title("Diagonal Values of D2", fontsize = 45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(3,2,5)
        pl.plot(np.array(self.SV_list), linewidth = 5)
        pl.grid()
        pl.title("Singular Value Check, Overall Matrix Rank: " + str(np.linalg.matrix_rank(P)) , fontsize = 45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(3,2,6)
        pl.plot(YforPlot, linewidth = 5)
        pl.title("Y last 25", fontsize = 45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())

    @staticmethod
    @njit
    def compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH):
        """_summary_

        Args:
            beta (_type_): _description_
            zeta (_type_): _description_
            D1 (_type_): _description_
            D2 (_type_): _description_
            M_H (_type_): _description_
            M_Y (_type_): _description_
            W_HX (_type_): _description_
            W_YH (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Mapping from xt -> ht
        A = (1 -zeta) * (beta * ((D1 * M_H) * D1.T) +  (1 - beta) * (M_H -  W_YH.T @ np.linalg.solve(M_Y, W_YH)))
        b = ((1 - zeta) * beta * (D1 * W_HX))
        WL1 = np.linalg.solve(A, b)
        # Mapping from ht -> yt
        WL2 = np.linalg.solve(M_Y * D2.T, W_YH)
        W = WL2 @ WL1
        return W

    def compute_overall_mapping(self, return_mapping = False):
        """_summary_

        Args:
            return_mapping (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH = self.beta, self.zeta, self.D1, self.D2, self.M_H, self.M_Y, self.W_HX, self.W_YH
        W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
        self.W = W 

        if return_mapping:
            return W

    def predict(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        beta, zeta, W_HX, W_YH, M_H, M_Y, D1, D2 = self.beta, self.zeta, self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
        return W @ X

    @staticmethod
    @njit
    def update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, D1, D2, 
                           MU_MH, MU_MY, MU_WHX, MU_WYH, muD, LayerMinimumGains, 
                           LayerMaximumGains, clip_gain_gradients = False):
        """_summary_

        Args:
            x_current (_type_): _description_
            h (_type_): _description_
            y (_type_): _description_
            zeta (_type_): _description_
            beta (_type_): _description_
            W_HX (_type_): _description_
            W_YH (_type_): _description_
            M_H (_type_): _description_
            M_Y (_type_): _description_
            D1 (_type_): _description_
            D2 (_type_): _description_
            MUS (_type_): _description_
            muD (_type_): _description_
            LayerMinimumGains (_type_): _description_
            LayerMaximumGains (_type_): _description_
        """
        def clipping(inp,lev):
            out=inp*(np.abs(inp)<=lev)+lev*(inp>lev)-lev*(inp<-lev)
            return out
        
        M_H = (1 - MU_MH) * M_H + MU_MH * np.outer(h,h)
        W_HX = (1 - MU_WHX) * W_HX + MU_WHX * np.outer(h,x_current)

        M_Y = (1 - MU_MY) * M_Y + MU_MY * np.outer(y,y)
        W_YH = (1 - MU_WYH) * W_YH + MU_WYH * np.outer(y,h)

        D1derivative = (1 - zeta) * beta * (np.sum((np.abs(M_H)**2) * D1.T,axis=1) - np.sum(np.abs(W_HX)**2,axis=1)).reshape(-1,1)  + zeta * (1/D1)#+ zeta * self.dlogdet(D1)
        if clip_gain_gradients:
            D1 = D1 - clipping(muD[0] * D1derivative, D1 * 1)
        else:
            D1 = D1 - muD[0] * D1derivative
        D1 = np.clip(D1, LayerMinimumGains[0], LayerMaximumGains[0])
        D2derivative = (1 - zeta) * (1 - beta) * (np.sum((np.abs(M_Y)**2) * D2.T,axis=1) - np.sum(np.abs(W_YH)**2,axis=1)).reshape(-1,1)  + zeta * (1/D2)#+ zeta * self.dlogdet(D2)
        if clip_gain_gradients:
            D2 = D2 - clipping(muD[1] * D2derivative, D2 * 1) 
        else:
            D2 = D2 - muD[1] * D2derivative
        D2 = np.clip(D2, LayerMinimumGains[1], LayerMaximumGains[1])
        return W_HX, W_YH, M_H, M_Y, D1, D2

    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                           neural_dynamic_iterations, lr_start, lr_stop, 
                                           lr_rule, lr_decay_multiplier = 0.01,
                                           hidden_layer_gain = 2, OUTPUT_COMP_TOL = 1e-7):
        """_summary_

        Args:
            x_current (_type_): _description_
            h (_type_): _description_
            y (_type_): _description_
            M_H (_type_): _description_
            M_Y (_type_): _description_
            W_HX (_type_): _description_
            W_YH (_type_): _description_
            D1 (_type_): _description_
            D2 (_type_): _description_
            beta (_type_): _description_
            zeta (_type_): _description_
            neural_dynamic_iterations (_type_): _description_
            lr_start (_type_): _description_
            lr_stop (_type_): _description_
            OUTPUT_COMP_TOL (_type_): _description_
        """
        # def ddiag(A):
        #     return np.diag(np.diag(A))
        
        def offdiag(A, return_diag = False):
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)
        
        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        
        v = mat_factor4[0] * h
        u = mat_factor6[0] * y
        
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*lr_decay_multiplier), lr_stop)

            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]
            y = np.clip(y, -1, 1)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                             neural_dynamic_iterations, lr_start, lr_stop, 
                                             lr_rule, lr_decay_multiplier = 0.01, hidden_layer_gain = 2, 
                                             use_hopfield = True, OUTPUT_COMP_TOL = 1e-7):
        """_summary_

        Args:
            x_current (_type_): _description_
            h (_type_): _description_
            y (_type_): _description_
            M_H (_type_): _description_
            M_Y (_type_): _description_
            W_HX (_type_): _description_
            W_YH (_type_): _description_
            D1 (_type_): _description_
            D2 (_type_): _description_
            beta (_type_): _description_
            zeta (_type_): _description_
            neural_dynamic_iterations (_type_): _description_
            lr_start (_type_): _description_
            lr_stop (_type_): _description_
            OUTPUT_COMP_TOL (_type_): _description_
        """
        def offdiag(A, return_diag = False):
            """_summary_

            Args:
                A (_type_): _description_
                return_diag (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)
        
        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        
        v = mat_factor4[0] * h
        u = mat_factor6[0] * y
        
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*lr_decay_multiplier), lr_stop)

            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]
            y = np.clip(y, 0, 1)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_sparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                       neural_dynamic_iterations, lr_start, lr_stop, 
                                       lr_rule, lr_decay_multiplier = 0.005, stlambd_lr = 1, 
                                       hidden_layer_gain = 100, mixtures_power_normalized = False, OUTPUT_COMP_TOL = 1e-7):
        def offdiag(A, return_diag = False):
            """_summary_

            Args:
                A (_type_): _description_
                return_diag (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)

        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded

        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        if mixtures_power_normalized:
            mat_factor6 = (1 - zeta) * Gamma_Y * ((1 - beta) + beta * D2.T ** 2)
            # mat_factor6 = (1 - zeta) * Gamma_Y * ((1 - beta) * D2.T ** 2)

        v = mat_factor4[0] * h
        u = mat_factor6[0] * y

        STLAMBD = 0
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*lr_decay_multiplier), lr_stop)

            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]
            y = SoftThresholding(y, STLAMBD)
            # y = np.clip(y, -1, 1)

            dval = np.linalg.norm(y,1) - 1
            
            STLAMBD = max(STLAMBD + stlambd_lr * dval,0)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                         neural_dynamic_iterations, lr_start, lr_stop, 
                                         lr_rule, lr_decay_multiplier = 0.005, stlambd_lr = 0.2, 
                                         hidden_layer_gain = 100, OUTPUT_COMP_TOL = 1e-7):
        def offdiag(A, return_diag = False):
            """_summary_

            Args:
                A (_type_): _description_
                return_diag (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)

        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        
        v = mat_factor4[0] * h
        u = mat_factor6[0] * y
        
        STLAMBD = 0
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*lr_decay_multiplier), lr_stop)

            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]
            y = np.maximum(y - STLAMBD, 0)
            # y = np.clip(y, 0, 1)

            dval = np.linalg.norm(y,1) - 1
            
            STLAMBD = max(STLAMBD + stlambd_lr * dval,0)

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_simplex_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                        neural_dynamic_iterations, lr_start, lr_stop, 
                                        lr_rule, lr_decay_multiplier = 0.005, stlambd_lr = 0.01, 
                                        hidden_layer_gain = 25, OUTPUT_COMP_TOL = 1e-7):

        def offdiag(A, return_diag = False):
            """_summary_

            Args:
                A (_type_): _description_
                return_diag (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)
        
        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        
        v = mat_factor4[0] * h
        u = mat_factor6[0] * y
        
        STLAMBD = 0
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*lr_decay_multiplier), lr_stop)
            # MUV = max(lr_start/(1+OutputCounter*0.005), lr_stop)
            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            # h = h*(h>=-2.0)*(h<=2.0)+(h>2.0)*2.0-2.0*(h<-2.0)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]

            y = np.maximum(y - STLAMBD, 0)
            dval = np.sum(y) - 1 
            STLAMBD = STLAMBD + stlambd_lr * dval

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter

    @staticmethod
    @njit
    def run_neural_dynamics_general_polytope_jit( x_current, h, y, signed_dims, nn_dims, sparse_dims_list,
                                                  M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                                  neural_dynamic_iterations, lr_start, lr_stop, 
                                                  lr_rule, lr_decay_multiplier = 0.005, stlambd_lr = 1, 
                                                  hidden_layer_gain = 100, OUTPUT_COMP_TOL = 1e-7):
        def offdiag(A, return_diag = False):
            """_summary_

            Args:
                A (_type_): _description_
                return_diag (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)

        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded

        def loop_intersection(lst1, lst2):
            result = []
            for element1 in lst1:
                for element2 in lst2:
                    if element1 == element2:
                        result.append(element1)
            return result

        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        
        v = mat_factor4[0] * h
        u = mat_factor6[0] * y
        
        STLAMBD_list = np.zeros(len(sparse_dims_list))
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*lr_decay_multiplier), lr_stop)

            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]
            if sparse_dims_list[0][0] != -1:
                for ss,sparse_dim in enumerate(sparse_dims_list):
                    # y[sparse_dim] = SoftThresholding(y[sparse_dim], STLAMBD_list[ss])
                    # STLAMBD_list[ss] = max(STLAMBD_list[ss] + (np.linalg.norm(y[sparse_dim],1) - 1), 0)
                    if signed_dims[0] != -1:
                        y[np.array(loop_intersection(sparse_dim, signed_dims))] = SoftThresholding(y[np.array(loop_intersection(sparse_dim, signed_dims))], STLAMBD_list[ss])
                    if nn_dims[0] != -1:
                        y[np.array(loop_intersection(sparse_dim, nn_dims))] = np.maximum(y[np.array(loop_intersection(sparse_dim, nn_dims))] - STLAMBD_list[ss], 0)
                    STLAMBD_list[ss] = max(STLAMBD_list[ss] + stlambd_lr * (np.linalg.norm(y[sparse_dim],1) - 1), 0)
            if signed_dims[0] != -1:
                y[signed_dims] = np.clip(y[signed_dims], -1, 1)
            if nn_dims[0] != -1:
                y[nn_dims] = np.clip(y[nn_dims], 0, 1)


            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter

    ###############################################################
    ######FIT NEXT FUNCTIONS FOR ONLINE LEARNING SETTING ##########
    ###############################################################

    def fit_next_antisparse(self, x_current, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, return_output = False):
        gamma_start, gamma_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gamma_start, self.gamma_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        
        i_sample = self.sample_index

        s_dim = self.s_dim
        h_dim = self.h_dim
        
        h = np.zeros((h_dim,))
        y = np.zeros((s_dim,))
        
        h,y, _ = self.run_neural_dynamics_antisparse_jit(   x_current = x_current, h = h, y = y, 
                                                            M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                            D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                            neural_dynamic_iterations = neural_dynamic_iterations, 
                                                            lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                            OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL  )

        MUS = np.max([gamma_start/(1+np.log(2 + i_sample)),gamma_stop])

        W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                D1, D2, MUS, muD, LayerMinimumGains, LayerMaximumGains )

        self.W_HX = W_HX
        self.W_YH = W_YH
        self.M_H = M_H
        self.M_Y = M_Y
        self.D1 = D1
        self.D2 = D2
        self.sample_index = i_sample + 1
        if return_output:
            return h,y
        else:
            return None, None


    ####################################################################
    ## FIT BATCH FUNCTIONS IF ALL THE OBSERVATIONS ARE AVAILABLE      ##
    ## THESE FUNCTIONS ALSO FIT IN ONLINE MANNER. YOU CAN CONSIDER    ##
    ## THEM AS EXTENSIONS OF FIT NEXT FUNCTIONS ABOVE (FOR DEBUGGING) ##
    ####################################################################
    def fit_batch_antisparse(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, 
                             synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                             synaptic_lr_decay_divider = 5000, neural_lr_decay_multiplier = 0.005,
                             hidden_layer_gain = 10, clip_gain_gradients = False, shuffle = True, debug_iteration_point = 1000,
                             plot_in_jupyter = False):

                            #  (self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, 
                            #    synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                            #    hidden_layer_gain = 10, clip_gain_gradients = True, shuffle = True, debug_iteration_point = 1000, 
                            #    plot_in_jupyter = False)
        """_summary_

        Args:
            X (_type_): _description_
            n_epochs (int, optional): _description_. Defaults to 5.
            neural_dynamic_iterations (int, optional): _description_. Defaults to 750.
            neural_lr_start (float, optional): _description_. Defaults to 0.2.
            neural_lr_stop (float, optional): _description_. Defaults to 0.05.
            shuffle (bool, optional): _description_. Defaults to True.
            debug_iteration_point (int, optional): _description_. Defaults to 1000.
            plot_in_jupyter (bool, optional): _description_. Defaults to False.
        """
        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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

                h,y, _ = self.run_neural_dynamics_antisparse_jit(x_current = x_current, h = h, y = y, 
                                                                 M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                                 D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                 neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                 lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                                 lr_rule = neural_loop_lr_rule, 
                                                                 lr_decay_multiplier = neural_lr_decay_multiplier,
                                                                 hidden_layer_gain = hidden_layer_gain,
                                                                 OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample//synaptic_lr_decay_divider + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample//synaptic_lr_decay_divider + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample//synaptic_lr_decay_divider + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample//synaptic_lr_decay_divider + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, S, X)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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

    def fit_batch_nnantisparse(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, 
                               synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                               synaptic_lr_decay_divider = 5000, neural_lr_decay_multiplier = 0.005,
                               hidden_layer_gain = 10, clip_gain_gradients = True, shuffle = True, debug_iteration_point = 1000, 
                               plot_in_jupyter = False):
        """_summary_

        Args:
            X (_type_): _description_
            n_epochs (int, optional): _description_. Defaults to 5.
            neural_dynamic_iterations (int, optional): _description_. Defaults to 750.
            neural_lr_start (float, optional): _description_. Defaults to 0.2.
            neural_lr_stop (float, optional): _description_. Defaults to 0.05.
            shuffle (bool, optional): _description_. Defaults to True.
            debug_iteration_point (int, optional): _description_. Defaults to 1000.
            plot_in_jupyter (bool, optional): _description_. Defaults to False.
        """
        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
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

                h,y, _ = self.run_neural_dynamics_nnantisparse_jit(x_current = x_current, h = h, y = y, 
                                                                   M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                                   D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                   neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                   lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                                   lr_rule = neural_loop_lr_rule, hidden_layer_gain = hidden_layer_gain,
                                                                   OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )
                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, S, X, False)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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

    def fit_batch_sparse(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, stlambd_lr = 1,
                         synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                         synaptic_lr_decay_divider = 5000, neural_lr_decay_multiplier = 0.005,
                         hidden_layer_gain = 10, clip_gain_gradients = False, mixtures_power_normalized = False, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):

        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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

                h,y, _ = self.run_neural_dynamics_sparse_jit(x_current = x_current, h = h, y = y, 
                                                             M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                             D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                             neural_dynamic_iterations = neural_dynamic_iterations, 
                                                             lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                             lr_rule = neural_loop_lr_rule, 
                                                             lr_decay_multiplier = neural_lr_decay_multiplier, stlambd_lr = stlambd_lr,
                                                             hidden_layer_gain = hidden_layer_gain, mixtures_power_normalized = mixtures_power_normalized,
                                                             OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, S, X)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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

    def fit_batch_nnsparse(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, stlambd_lr = 0.2,
                           synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                           synaptic_lr_decay_divider = 5000, neural_lr_decay_multiplier = 0.005,
                           hidden_layer_gain = 10, clip_gain_gradients = False, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):

        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
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

                h,y, _ = self.run_neural_dynamics_nnsparse_jit(x_current = x_current, h = h, y = y, 
                                                               M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                               D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                               neural_dynamic_iterations = neural_dynamic_iterations, 
                                                               lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                               lr_rule = neural_loop_lr_rule, lr_decay_multiplier = neural_lr_decay_multiplier,
                                                               stlambd_lr = stlambd_lr, hidden_layer_gain = hidden_layer_gain, 
                                                               OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, Szeromean, X, mean_normalize_estimation = True)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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

    def fit_batch_simplex(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05,  stlambd_lr = 0.05,
                          synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                          synaptic_lr_decay_divider = 5000, neural_lr_decay_multiplier = 0.005,
                          hidden_layer_gain = 25, clip_gain_gradients = False, shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        """_summary_

        Args:
            X (_type_): _description_
            n_epochs (int, optional): _description_. Defaults to 5.
            neural_dynamic_iterations (int, optional): _description_. Defaults to 750.
            neural_lr_start (float, optional): _description_. Defaults to 0.2.
            neural_lr_stop (float, optional): _description_. Defaults to 0.05.
            shuffle (bool, optional): _description_. Defaults to True.
            debug_iteration_point (int, optional): _description_. Defaults to 1000.
            plot_in_jupyter (bool, optional): _description_. Defaults to False.
        """
        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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
            Szeromean = S - S.mean(axis = 1).reshape(-1,1) 
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

                h,y, _ = self.run_neural_dynamics_simplex_jit(  x_current = x_current, h = h, y = y, 
                                                                M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                                D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                                lr_rule = neural_loop_lr_rule, lr_decay_multiplier = neural_lr_decay_multiplier,
                                                                stlambd_lr = stlambd_lr, hidden_layer_gain = hidden_layer_gain, 
                                                                OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)
                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, Szeromean, X, mean_normalize_estimation = True)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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

    def fit_batch_general_polytope(self, X, signed_dims, nn_dims, sparse_dims_list, n_epochs = 5, neural_dynamic_iterations = 750, 
                                   neural_lr_start = 0.2, neural_lr_stop = 0.05, stlambd_lr = 1,
                                   synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                                   synaptic_lr_decay_divider = 5000, neural_lr_decay_multiplier = 0.005,
                                   hidden_layer_gain = 10, clip_gain_gradients = False, shuffle = True, 
                                   debug_iteration_point = 1000, plot_in_jupyter = False):

        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
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
                
                if ((i_sample + 1) % 100000) == 0:
                    muD = 0.99 * np.array(muD)
                
                x_current  = X[:,idx[i_sample]] # Take one input

                y = Y[:,idx[i_sample]]

                h = H[:,idx[i_sample]]
                neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL

                h,y, _ = self.run_neural_dynamics_general_polytope_jit( x_current = x_current, h = h, y = y, 
                                                                        signed_dims = signed_dims, nn_dims = nn_dims, 
                                                                        sparse_dims_list = sparse_dims_list,
                                                                        M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                                        D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                                        neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                        lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                                        lr_rule = neural_loop_lr_rule, 
                                                                        lr_decay_multiplier = neural_lr_decay_multiplier,
                                                                        stlambd_lr = stlambd_lr,
                                                                        hidden_layer_gain = hidden_layer_gain, 
                                                                        OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//synaptic_lr_decay_divider))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, Szeromean, X, mean_normalize_estimation = True)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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


class OnlineWSMICA(OnlineWSMBSS):
    #### ONLY WRITTEN TO TRY. NO SUCCESS YET
    @staticmethod
    @njit
    def run_neural_dynamics_ica_jit(x_current, h, y, M_H, M_Y, W_HX, W_YH, D1, D2, beta, zeta, 
                                    neural_dynamic_iterations, lr_start, lr_stop, 
                                    lr_rule, hidden_layer_gain = 20, output_activation = "tanh",
                                    OUTPUT_COMP_TOL = 1e-7):
        """_summary_

        Args:
            x_current (_type_): _description_
            h (_type_): _description_
            y (_type_): _description_
            M_H (_type_): _description_
            M_Y (_type_): _description_
            W_HX (_type_): _description_
            W_YH (_type_): _description_
            D1 (_type_): _description_
            D2 (_type_): _description_
            beta (_type_): _description_
            zeta (_type_): _description_
            neural_dynamic_iterations (_type_): _description_
            lr_start (_type_): _description_
            lr_stop (_type_): _description_
            OUTPUT_COMP_TOL (_type_): _description_
        """
        def offdiag(A, return_diag = False):
            """_summary_

            Args:
                A (_type_): _description_
                return_diag (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            if return_diag:
                diag = np.diag(A)
                return A - np.diag(diag), diag
            else:
                return A - np.diag(diag)
        
        M_hat_H, Gamma_H = offdiag(M_H, True)
        M_hat_Y, Gamma_Y = offdiag(M_Y, True)
        
        mat_factor1 = (1 - zeta) * beta * (D1 * W_HX)
        mat_factor2 = ((1 - zeta) * (1 - beta) * M_hat_H  + (1- zeta) * beta * ((D1 * M_hat_H) * D1.T))
        mat_factor3 = (1 - zeta) * (1 - beta) * (W_YH.T * D2.T)
        mat_factor4 = (1 - zeta) * Gamma_H * ((1 - beta) + beta * (D1.T) ** 2)
        mat_factor5 = M_hat_Y * D2.T
        mat_factor6 = Gamma_Y * D2.T
        
        v = mat_factor4[0] * h
        u = mat_factor6[0] * y
        
        PreviousMembraneVoltages = {'v': np.zeros_like(v), 'u': np.zeros_like(u)}
        MembraneVoltageNotSettled = 1
        OutputCounter = 0
        while MembraneVoltageNotSettled & (OutputCounter < neural_dynamic_iterations):
            OutputCounter += 1
            if lr_rule == "constant":
                MUV = lr_start
            elif lr_rule == "divide_by_loop_index":
                MUV = max(lr_start/(1+OutputCounter), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                MUV = max(lr_start/(1+OutputCounter*0.05), lr_stop)

            delv = -v + mat_factor1 @ x_current - mat_factor2 @ h + mat_factor3 @ y
            v = v + MUV * delv
            h = v / mat_factor4[0]
            h = np.clip(h, -hidden_layer_gain, hidden_layer_gain)
            delu = -u + W_YH @ h - mat_factor5 @ y
            u = u + MUV * delu
            y = u / mat_factor6[0]
            if output_activation == "tanh":
                y = np.tanh(y)
            elif output_activation == "linear":
                y = y

            MembraneVoltageNotSettled = 0
            if (np.linalg.norm(v - PreviousMembraneVoltages['v'])/np.linalg.norm(v) > OUTPUT_COMP_TOL) | (np.linalg.norm(u - PreviousMembraneVoltages['u'])/np.linalg.norm(u) > OUTPUT_COMP_TOL):
                MembraneVoltageNotSettled = 1
            PreviousMembraneVoltages['v'] = v
            PreviousMembraneVoltages['u'] = u

        return h,y, OutputCounter


    def fit_batch_ica(self, X, n_epochs = 5, neural_dynamic_iterations = 750, neural_lr_start = 0.2, neural_lr_stop = 0.05, 
                      synaptic_lr_rule = "divide_by_log_index", neural_loop_lr_rule = "divide_by_slow_loop_index", 
                      hidden_layer_gain = 2, clip_gain_gradients = True, output_activation = "tanh", 
                      shuffle = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        """_summary_

        Args:
            X (_type_): _description_
            n_epochs (int, optional): _description_. Defaults to 5.
            neural_dynamic_iterations (int, optional): _description_. Defaults to 750.
            neural_lr_start (float, optional): _description_. Defaults to 0.2.
            neural_lr_stop (float, optional): _description_. Defaults to 0.05.
            shuffle (bool, optional): _description_. Defaults to True.
            debug_iteration_point (int, optional): _description_. Defaults to 1000.
            plot_in_jupyter (bool, optional): _description_. Defaults to False.
        """
        gammaM_start, gammaM_stop, gammaW_start, gammaW_stop, beta, zeta, muD, W_HX, W_YH, M_H, M_Y, D1, D2 = self.gammaM_start, self.gammaM_stop, self.gammaW_start, self.gammaW_stop, self.beta, self.zeta, np.array(self.muD), self.W_HX, self.W_YH, self.M_H, self.M_Y, self.D1, self.D2
        LayerMinimumGains = self.LayerMinimumGains
        LayerMaximumGains = self.LayerMaximumGains
        debugging = self.set_ground_truth
    
        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        D1list = []
        D2list = []
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
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
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

                h,y, _ = self.run_neural_dynamics_ica_jit(  x_current = x_current, h = h, y = y, 
                                                            M_H = M_H, M_Y = M_Y, W_HX = W_HX, W_YH = W_YH, 
                                                            D1 = D1, D2 = D2, beta = beta, zeta = zeta, 
                                                            neural_dynamic_iterations = neural_dynamic_iterations, 
                                                            lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                            lr_rule = neural_loop_lr_rule, hidden_layer_gain = hidden_layer_gain,
                                                            output_activation = output_activation, OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL)

                if synaptic_lr_rule == "constant":
                    MU_MH = gammaM_start[0]
                    MU_MY = gammaM_start[1]
                    MU_WHX = gammaW_start[0]
                    MU_WYH = gammaW_start[1]
                elif synaptic_lr_rule == "divide_by_log_index":
                    # MUS = np.max([gamma_start/(1 + np.log(2 + (i_sample))), gamma_stop])
                    MU_MH = np.max([gammaM_start[0]/(1 + np.log(2 + (i_sample//5000))), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(1 + np.log(2 + (i_sample//5000))), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(1 + np.log(2 + (i_sample//5000))), gammaW_stop[0]])
                    MU_WYH = np.max([gammaW_start[1]/(1 + np.log(2 + (i_sample//5000))), gammaW_stop[1]])
                elif synaptic_lr_rule == "divide_by_index":
                    MU_MH = np.max([gammaM_start[0]/(i_sample + 1), gammaM_stop[0]])
                    MU_MY = np.max([gammaM_start[1]/(i_sample + 1), gammaM_stop[1]])
                    MU_WHX = np.max([gammaW_start[0]/(i_sample + 1), gammaW_stop[0]])
                    MU_WYH= np.max([gammaW_start[1]/(i_sample + 1), gammaW_stop[1]])

                W_HX, W_YH, M_H, M_Y, D1, D2 = self.update_weights_jit(x_current, h, y, zeta, beta, W_HX, W_YH, M_H, M_Y, 
                                                                       D1, D2, MU_MH, MU_MY, MU_WHX, MU_WYH, muD, 
                                                                       LayerMinimumGains, LayerMaximumGains, clip_gain_gradients )

                Y[:,idx[i_sample]] = y
                H[:,idx[i_sample]] = h

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            W = self.compute_overall_mapping_jit(beta, zeta, D1, D2, M_H, M_Y, W_HX, W_YH)
                            self.W = W
                            
                            SINR_current, SNR_current, SGG, Y_, P = self.evaluate_for_debug(W, A, Szeromean, X, mean_normalize_estimation = True)

                            self.SV_list.append(abs(SGG))

                            SNR_list.append(SNR_current)
                            SIR_list.append(SINR_current)

                            if plot_in_jupyter:
                                D1list.append(D1.reshape(-1,))
                                D2list.append(D2.reshape(-1,))
                                YforPlot = Y[:,idx[i_sample-25:i_sample]].T
                                self.plot_for_debug(SIR_list, SNR_list, D1list, D2list, P, debug_iteration_point, YforPlot)

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

