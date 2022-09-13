import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from IPython.display import Latex, Math, clear_output, display
from numba import njit
from tqdm import tqdm

from bss_utils import *
##### IMPORT MY UTILITY SCRIPTS #######
from BSSbase import *
from dsp_utils import *
# from general_utils import *
from numba_utils import *

# from visualization_utils import *


class PMFBaseClass(BSSBaseClass):
    ###################################################################
    ############### FUNCTIONS FOR DEBUGGING ###########################
    ###################################################################
    def evaluate_for_debug(self, W, Y, A, S):
        s_dim = self.s_dim
        # Y = Y - Y.mean(axis = 1, keepdims = True)
        # S = S - S.mean(axis = 1, keepdims = True)
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
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
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
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
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
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
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


class PMF(PMFBaseClass):

    """
    Implementation of batch Polytopic Matrix Factorization
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    y_dim          -- Dimension of the mixtures
    H              -- Feedforward Synapses

    Methods:
    ==================================

    """

    def __init__(
        self, s_dim, y_dim, H=None, set_ground_truth=False, Sgt=None, Agt=None
    ):
        if H is not None:
            assert H.shape == (
                y_dim,
                s_dim,
            ), "The shape of the initial guess H must be (s_dim, y_dim) = (%d,%d)" % (
                s_dim,
                y_dim,
            )
            H = H
        else:
            H = np.random.randn(y_dim, s_dim)

        self.s_dim = s_dim
        self.y_dim = y_dim
        self.H = H

        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.Sgt = Sgt  # Ground Truth Sources
        self.Agt = Agt  # Ground Truth Mixing Matrix
        self.SIR_list = []
        self.SINR_list = []
        self.SNR_list = []

    def fit_batch_antisparse(
        self,
        Y,
        n_iterations=10000,
        Lt=50,
        lambda_=25,
        tau=1e-10,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        S = V[0 : self.s_dim, :]
        H = U[:, : self.s_dim]

        Identity = np.eye(self.s_dim)
        F = Identity.copy()
        q = 1

        diff = S.copy()
        for k in tqdm(range(n_iterations)):
            #### PMF Algorithm #################
            Sprev = S.copy()
            S = S + ((q - 1) / ((1 + math.sqrt(1 + 4 * q * q)) / 2)) * (diff)
            S = S - (
                np.dot(H.T, (np.dot(H, S) - Y))
                / (Lt * np.linalg.norm(np.transpose(H) @ H, 2))
            )
            q = (1 + math.sqrt(1 + 4 * q * q)) / 2
            S = np.clip(S, -1, 1)
            diff = S - Sprev

            H = np.dot(np.dot(Y, S.T), np.linalg.inv(np.dot(S, S.T) + lambda_ * F))

            F = np.linalg.inv(np.dot(np.transpose(H), H) + tau * Identity)
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = H
                        W = np.linalg.pinv(H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        self.H = H
        self.S = S

    def fit_batch_nnantisparse(
        self,
        Y,
        n_iterations=10000,
        Lt=50,
        lambda_=25,
        tau=1e-10,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        S = V[0 : self.s_dim, :]
        H = U[:, : self.s_dim]

        Identity = np.eye(self.s_dim)
        F = Identity.copy()
        q = 1

        diff = S.copy()
        for k in tqdm(range(n_iterations)):
            #### PMF Algorithm #################
            Sprev = S.copy()
            S = S + ((q - 1) / ((1 + math.sqrt(1 + 4 * q * q)) / 2)) * (diff)
            S = S - (
                np.dot(H.T, (np.dot(H, S) - Y))
                / (Lt * np.linalg.norm(np.transpose(H) @ H, 2))
            )
            q = (1 + math.sqrt(1 + 4 * q * q)) / 2
            S = np.clip(S, 0, 1)
            diff = S - Sprev

            H = np.dot(np.dot(Y, S.T), np.linalg.inv(np.dot(S, S.T) + lambda_ * F))

            F = np.linalg.inv(np.dot(np.transpose(H), H) + tau * Identity)
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = H
                        W = np.linalg.pinv(H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        self.H = H
        self.S = S

    def fit_batch_simplex(
        self,
        Y,
        n_iterations=10000,
        Lt=50,
        lambda_=25,
        tau=1e-10,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        S = V[0 : self.s_dim, :]
        H = U[:, : self.s_dim]

        Identity = np.eye(self.s_dim)
        F = Identity.copy()
        q = 1

        diff = S.copy()
        for k in tqdm(range(n_iterations)):
            #### PMF Algorithm #################
            Sprev = S.copy()
            S = S + ((q - 1) / ((1 + math.sqrt(1 + 4 * q * q)) / 2)) * (diff)
            S = S - (
                np.dot(H.T, (np.dot(H, S) - Y))
                / (Lt * np.linalg.norm(np.transpose(H) @ H, 2))
            )
            q = (1 + math.sqrt(1 + 4 * q * q)) / 2
            S = self.ProjectColstoSimplex(S)
            diff = S - Sprev

            H = np.dot(np.dot(Y, S.T), np.linalg.inv(np.dot(S, S.T) + lambda_ * F))

            F = np.linalg.inv(np.dot(np.transpose(H), H) + tau * Identity)
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = H
                        W = np.linalg.pinv(H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        self.H = H
        self.S = S

    def fit_batch_general_polytope(
        self,
        Y,
        signed_dims,
        nn_dims,
        sparse_dims_list,
        n_iterations=10000,
        Lt=50,
        lambda_=25,
        tau=1e-10,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        S = V[0 : self.s_dim, :]
        H = U[:, : self.s_dim]

        Identity = np.eye(self.s_dim)
        F = Identity.copy()
        q = 1

        diff = S.copy()
        for k in tqdm(range(n_iterations)):
            #### PMF Algorithm #################
            Sprev = S.copy()
            S = S + ((q - 1) / ((1 + math.sqrt(1 + 4 * q * q)) / 2)) * (diff)
            S = S - (
                np.dot(H.T, (np.dot(H, S) - Y))
                / (Lt * np.linalg.norm(np.transpose(H) @ H, 2))
            )
            q = (1 + math.sqrt(1 + 4 * q * q)) / 2
            S = self.ProjectColumnsOntoPracticalPolytope(
                S, signed_dims, nn_dims, sparse_dims_list
            )
            diff = S - Sprev

            H = np.dot(np.dot(Y, S.T), np.linalg.inv(np.dot(S, S.T) + lambda_ * F))

            F = np.linalg.inv(np.dot(np.transpose(H), H) + tau * Identity)
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = H
                        W = np.linalg.pinv(H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        self.H = H
        self.S = S


class PMFv2(PMFBaseClass):
    """
    Implementation of batch Polytopic Matrix Factorization
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    y_dim          -- Dimension of the mixtures
    H              -- Feedforward Synapses

    Methods:
    ==================================

    """

    def __init__(
        self, s_dim, y_dim, H=None, set_ground_truth=False, Sgt=None, Agt=None
    ):
        if H is not None:
            assert H.shape == (
                y_dim,
                s_dim,
            ), "The shape of the initial guess H must be (s_dim, y_dim) = (%d,%d)" % (
                s_dim,
                y_dim,
            )
            H = H
        else:
            H = np.random.randn(y_dim, s_dim)

        self.s_dim = s_dim
        self.y_dim = y_dim
        self.H = H

        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.Sgt = Sgt  # Ground Truth Sources
        self.Agt = Agt  # Ground Truth Mixing Matrix
        self.SIR_list = []
        self.SINR_list = []
        self.SNR_list = []

    def fit_batch_antisparse(
        self,
        Y,
        n_iterations=10000,
        step_size_scale=100,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        debugging = self.set_ground_truth

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        SS = np.diag(singulars)

        S = V[0 : self.s_dim, :]
        R = (S.T).copy()
        B = R.T @ S.T
        for k in tqdm(range(n_iterations)):
            muk = step_size_scale / np.sqrt(k + 1.0)
            B = B + muk * np.linalg.inv(B).T
            RR = B.T @ R.T
            S = np.clip(RR, -1, 1)
            B = R.T @ S.T
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = (
                            U[:, 0 : self.s_dim]
                            @ SS[0 : self.s_dim, 0 : self.s_dim]
                            @ np.linalg.inv(B).T
                        )
                        W = np.linalg.pinv(self.H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        H = (
            U[:, 0 : self.s_dim]
            @ SS[0 : self.s_dim, 0 : self.s_dim]
            @ np.linalg.inv(B).T
        )
        self.H = H
        self.S = S
        W = np.linalg.pinv(self.H)
        self.W = W

    def fit_batch_nnantisparse(
        self,
        Y,
        n_iterations=10000,
        step_size_scale=100,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        debugging = self.set_ground_truth

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        SS = np.diag(singulars)

        S = V[0 : self.s_dim, :]
        R = (S.T).copy()
        Stemp = S.copy()
        for kk in range(self.s_dim):
            mv = np.max(np.abs(Stemp[:, kk]))
            mvi = np.asarray(np.where(np.abs(Stemp[:, kk]) == mv))
            # U[:,kk] = U[:,kk] * np.sign(Stemp[mvi[0,0],kk])
            S[:, kk] = Stemp[:, kk] * np.sign(Stemp[mvi[0, 0], kk])

        S = S / np.max(S) / 2
        S = np.clip(S, 0, 1)

        B = R.T @ S.T
        for k in tqdm(range(n_iterations)):
            muk = step_size_scale / np.sqrt(k + 1.0)
            B = B + muk * np.linalg.inv(B).T
            RR = B.T @ R.T
            S = np.clip(RR, 0, 1)
            B = R.T @ S.T
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = (
                            U[:, 0 : self.s_dim]
                            @ SS[0 : self.s_dim, 0 : self.s_dim]
                            @ np.linalg.inv(B).T
                        )
                        W = np.linalg.pinv(self.H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        H = (
            U[:, 0 : self.s_dim]
            @ SS[0 : self.s_dim, 0 : self.s_dim]
            @ np.linalg.inv(B).T
        )
        self.H = H
        W = np.linalg.pinv(self.H)
        self.W = W
        self.S = S

    def fit_batch_sparse(
        self,
        Y,
        n_iterations=10000,
        step_size_scale=100,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        debugging = self.set_ground_truth

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        SS = np.diag(singulars)

        S = V[0 : self.s_dim, :]
        R = (S.T).copy()
        B = R.T @ S.T
        for k in tqdm(range(n_iterations)):
            muk = step_size_scale / np.sqrt(k + 1.0)
            B = B + muk * np.linalg.inv(B).T
            RR = B.T @ R.T
            S = self.ProjectRowstoL1NormBall(RR.T).T
            B = R.T @ S.T
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = (
                            U[:, 0 : self.s_dim]
                            @ SS[0 : self.s_dim, 0 : self.s_dim]
                            @ np.linalg.inv(B).T
                        )
                        W = np.linalg.pinv(self.H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        H = (
            U[:, 0 : self.s_dim]
            @ SS[0 : self.s_dim, 0 : self.s_dim]
            @ np.linalg.inv(B).T
        )
        self.H = H
        self.S = S

    def fit_batch_nnsparse(
        self,
        Y,
        n_iterations=10000,
        step_size_scale=100,
        debug_iteration_point=100,
        plot_in_jupyter=False,
    ):

        debugging = self.set_ground_truth
        samples = Y.shape[1]
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            self.SV_list = []
            Sgt = self.Sgt
            Agt = self.Agt
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        debugging = self.set_ground_truth

        U, singulars, V = np.linalg.svd(Y, full_matrices=False)
        SS = np.diag(singulars)

        S = V[0 : self.s_dim, :]
        R = (S.T).copy()
        Stemp = S.copy()
        for kk in range(self.s_dim):
            mv = np.max(np.abs(Stemp[:, kk]))
            mvi = np.asarray(np.where(np.abs(Stemp[:, kk]) == mv))
            # U[:,kk] = U[:,kk] * np.sign(Stemp[mvi[0,0],kk])
            S[:, kk] = Stemp[:, kk] * np.sign(Stemp[mvi[0, 0], kk])

        S = np.clip(self.ProjectRowstoL1NormBall(S.T).T, 0, 1)

        B = R.T @ S.T
        for k in tqdm(range(n_iterations)):
            muk = step_size_scale / np.sqrt(k + 1.0)
            B = B + muk * np.linalg.inv(B).T
            RR = B.T @ R.T
            S = np.clip(self.ProjectRowstoL1NormBall(RR.T).T, 0, 1)
            B = R.T @ S.T
            if debugging:
                if ((k % debug_iteration_point) == 0) | (k == n_iterations - 1):
                    try:
                        self.H = (
                            U[:, 0 : self.s_dim]
                            @ SS[0 : self.s_dim, 0 : self.s_dim]
                            @ np.linalg.inv(B).T
                        )
                        W = np.linalg.pinv(self.H)
                        self.W = W
                        SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(W, S, Agt, Sgt)
                        self.SV_list.append(abs(SGG))
                        SIR_list.append(SINR)
                        SNR_list.append(SNR)

                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list

                        if plot_in_jupyter:
                            random_idx = np.random.randint(S.shape[1] - 25)
                            YforPlot = S[:, random_idx - 25 : random_idx].T
                            self.plot_for_debug(
                                SIR_list, SNR_list, P, debug_iteration_point, YforPlot
                            )
                    except Exception as e:
                        print(str(e))
        H = (
            U[:, 0 : self.s_dim]
            @ SS[0 : self.s_dim, 0 : self.s_dim]
            @ np.linalg.inv(B).T
        )
        self.H = H
        self.S = S
