import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from IPython.display import Latex, Math, clear_output, display
from numba import njit
from tqdm import tqdm

##### IMPORT MY UTILITY SCRIPTS #######
from BSSbase import *

# from dsp_utils import *
# from bss_utils import *
# # from general_utils import *
# from numba_utils import *
# from visualization_utils import *


class RVolMin(BSSBaseClass):
    ###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #####################
    ###### STILL UNDER DEVELOPMENT, MIGHT NOT WORKING PROPERLY YET ###############
    """
    Implementation of 'Robust Volume Minimization-Based Matrix Factorization for Remote
    Sensing and Document Clustering'
    Parameters:
    =================================


    Methods:
    ==================================


    """

    def __init__(self, s_dim, x_dim, set_ground_truth=False, S=None, A=None) -> None:
        super().__init__()

        self.s_dim = s_dim
        self.x_dim = x_dim
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S  # Sources
        self.A = A  # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []

    ###################################################################
    ############### FUNCTIONS FOR DEBUGGING ###########################
    ###################################################################
    def evaluate_for_debug(self, W, Y, A, S):
        s_dim = self.s_dim
        Y = Y - Y.mean(axis=1, keepdims=True)
        S = S - S.mean(axis=1, keepdims=True)
        Y_ = self.signed_and_permutation_corrected_sources(S, Y)
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
        Y_ = coef_ * Y_

        SINR = 10 * np.log10(self.CalculateSINRjit(Y_, S, False)[0])
        SNR = self.snr_jit(S, Y_)

        T = W @ A
        Tabs = np.abs(T)
        P = np.zeros((s_dim, s_dim))

        for SourceIndex in range(s_dim):
            Tmax = np.max(Tabs[SourceIndex, :])
            Tabs[SourceIndex, :] = Tabs[SourceIndex, :] / Tmax
            P[SourceIndex, :] = Tabs[SourceIndex, :] > 0.999

        GG = P.T @ T
        _, SGG, _ = np.linalg.svd(
            GG
        )  # SGG is the singular values of overall matrix Wf @ A

        return SINR, SNR, SGG, Y_, P

    def plot_for_debug(self, SIR_list, SNR_list, P, debug_iteration_point, YforPlot):
        pl.clf()
        pl.subplot(2, 2, 1)
        pl.plot(np.array(SIR_list), linewidth=5)
        pl.xlabel(f"Number of Iterations / {debug_iteration_point}", fontsize=45)
        pl.ylabel("SIR (dB)", fontsize=45)
        pl.title("SIR Behaviour", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 2)
        pl.plot(np.array(SNR_list), linewidth=5)
        pl.grid()
        pl.title("Component SNR Check", fontsize=45)
        pl.ylabel("SNR (dB)", fontsize=45)
        pl.xlabel(f"Number of Iterations / {debug_iteration_point}", fontsize=45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 3)
        pl.plot(np.array(self.SV_list), linewidth=5)
        pl.grid()
        pl.title(
            "Singular Value Check, Overall Matrix Rank: "
            + str(np.linalg.matrix_rank(P)),
            fontsize=45,
        )
        pl.xlabel(f"Number of Iterations / {debug_iteration_point}", fontsize=45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 4)
        pl.plot(YforPlot, linewidth=5)
        pl.title("Random 25 Output (from Y)", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())

    def fit_batch(
        self,
        X,
        n_iterations=10000,
        p=0.5,
        Lt=50,
        lambda_=1,
        tau=1e-8,
        epsilon=1e-12,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = X.shape[1]

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        U, singulars, V = np.linalg.svd(X, full_matrices=False)
        C = V[0 : self.s_dim, :]
        B = U[:, : self.s_dim]
        Identity = np.eye(self.s_dim)
        F = Identity.copy()
        W = np.ones((samples, 1))
        q = 1

        diff = C.copy()
        Y = C.copy()
        for k in tqdm(range(n_iterations)):
            Cprev = C.copy()
            q_prev = q
            q = (1 + math.sqrt(1 + 4 * q * q)) / 2
            C = C + ((q_prev - 1) / q) * (diff)
            C = C - (
                np.dot(B.T, (np.dot(B, C) - X)) / Lt
            )  # (Lt*np.linalg.norm(np.transpose(B)@B, 2)))
            C = self.ProjectColstoSimplex(C)
            diff = C - Cprev
            B = (X * W.T) @ C.T @ np.linalg.pinv((C * W.T) @ C.T + lambda_ * F)
            F = np.linalg.inv(np.dot(np.transpose(B), B) + tau * Identity)
            W = (
                (p / 2) * (np.sum((X - B @ C) ** 2, axis=0) + epsilon) ** ((p - 2) / 2)
            )[:, np.newaxis]
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.B = B
                        Wseparator = np.linalg.pinv(B)
                        self.Wseparator = Wseparator
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                            Wseparator, S, A, C
                        )
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(C.shape[1] - 25)
                            CforPlot = C[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, CforPlot
                            )
                    except Exception as e:
                        print(str(e))
