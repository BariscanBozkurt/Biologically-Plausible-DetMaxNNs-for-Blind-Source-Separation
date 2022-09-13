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

mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15


class OnlineNSM(BSSBaseClass):
    """Implementation of Online Nonnegative Similarity Matching.

    Parameters:
    ==========================================================
    s_dim           --Dimension of the sources
    x_dim           --Dimension of the mixtures
    W1
    W2
    Dt
    neural_OUTPUT_COMP_TOL
    set_ground_truth
    S               --Original Sources (for debugging only)
    A               --Mixing Matrix (for debugging only)

    Methods:
    ===========================================================
    whiten_input
    snr
    compute_overall_mapping
    CalculateSIR
    predict
    run_neural_dynamics
    fit_batch_nsm
    """

    def __init__(
        self,
        s_dim,
        x_dim,
        W1=None,
        W2=None,
        Dt=None,
        whiten_input_=True,
        set_ground_truth=False,
        S=None,
        A=None,
    ):

        if W1 is not None:
            if whiten_input_:
                assert W1.shape == (s_dim, s_dim), (
                    "The shape of the initial guess W1 must be (s_dim, s_dim) = (%d, %d)"
                    % (s_dim, s_dim)
                )
                W1 = W1
            else:
                assert W1.shape == (s_dim, x_dim), (
                    "The shape of the initial guess W1 must be (s_dim, x_dim) = (%d, %d)"
                    % (s_dim, x_dim)
                )
                W1 = W1
        else:
            if whiten_input_:
                W1 = np.eye(s_dim, s_dim)
            else:
                W1 = np.eye(s_dim, x_dim)

        if W2 is not None:
            assert W2.shape == (s_dim, s_dim), (
                "The shape of the initial guess W2 must be (s_dim, s_dim) = (%d, %d)"
                % (s_dim, s_dim)
            )
            W2 = W2
        else:
            W2 = np.zeros((s_dim, s_dim))

        if Dt is not None:
            assert Dt.shape == (
                s_dim,
                1,
            ), "The shape of the initial guess Dt must be (s_dim, 1) = (%d, %d)" % (
                s_dim,
                1,
            )
            Dt = Dt
        else:
            Dt = 0.1 * np.ones((s_dim, 1))

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W1 = W1
        self.W2 = W2
        self.Dt = Dt
        self.whiten_input_ = whiten_input_
        self.Wpre = np.eye(x_dim)
        self.set_ground_truth = set_ground_truth
        self.S = S
        self.A = A
        self.SIR_list = []
        self.SNR_list = []

    def whiten_input(self, X):
        x_dim = self.x_dim
        s_dim = self.s_dim
        N = X.shape[1]
        # Mean of the mixtures
        mX = np.mean(X, axis=1).reshape((x_dim, 1))
        # Covariance of Mixtures
        Rxx = np.dot(X, X.T) / N - np.dot(mX, mX.T)
        # Eigenvalue Decomposition
        d, V = np.linalg.eig(Rxx)
        D = np.diag(d)
        # Sorting indexis for eigenvalues from large to small
        ie = np.argsort(-d)
        # Inverse square root of eigenvalues
        ddinv = 1 / np.sqrt(d[ie[:s_dim]])
        # Pre-whitening matrix
        Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)  # *np.sqrt(12)
        # Whitened mixtures
        H = np.dot(Wpre, X)
        self.Wpre = Wpre
        return H

    def compute_overall_mapping(self, return_mapping=False):
        W1, W2 = self.W1, self.W2
        Wpre = self.Wpre
        # W = np.linalg.pinv(np.eye(self.s_dim) + W2) @ W1 @ Wpre
        W = np.linalg.solve(np.eye(self.s_dim) + W2, W1) @ Wpre
        self.W = W
        if return_mapping:
            return W

    def evaluate_for_debug(self, W, A, S, X, mean_normalize_estimation=False):
        s_dim = self.s_dim
        Y_ = W @ X
        if mean_normalize_estimation:
            Y_ = Y_ - Y_.mean(axis=1).reshape(-1, 1)
        Y_ = self.signed_and_permutation_corrected_sources(S, Y_)
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
        pl.title("Y last 25", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())

    def predict(self, X):
        Wf = self.compute_overall_mapping(return_mapping=True)
        return Wf @ X

    @staticmethod
    @njit
    def run_neural_dynamics(x, y, W1, W2, n_iterations=200):
        for j in range(n_iterations):
            ind = math.floor((np.random.rand(1) * y.shape[0])[0])
            y[ind, :] = np.maximum(np.dot(W1[ind, :], x) - np.dot(W2[ind, :], y), 0)

        return y

    def fit_batch_nsm(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        shuffle=True,
        debug_iteration_point=10000,
        plot_in_jupyter=False,
    ):
        s_dim, x_dim = self.s_dim, self.x_dim
        W1, W2, Dt = self.W1, self.W2, self.Dt
        debugging = self.set_ground_truth
        ZERO_CHECK_INTERVAL = 1500
        nzerocount = np.zeros(s_dim)
        whiten_input_ = self.whiten_input_
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        assert (
            X.shape[0] == self.x_dim
        ), "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        Y = np.random.rand(self.s_dim, samples)

        if whiten_input_:
            X_ = self.whiten_input(X)
            x_dim = X_.shape[0]
        else:
            X_ = X

        Wpre = self.Wpre

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)

            for i_sample in tqdm(range(samples)):
                x_current = X_[:, idx[i_sample]]
                xk = np.reshape(x_current, (-1, 1))

                y = np.random.rand(s_dim, 1)

                y = self.run_neural_dynamics(xk, y, W1, W2, neural_dynamic_iterations)

                # Record the seperated signal
                Y[:, idx[i_sample]] = y.reshape(
                    -1,
                )

                # Update Weights
                Dt = np.minimum(3000, 0.94 * Dt + y**2)
                DtD = np.diag(1 / Dt.reshape(s_dim))
                W1 = W1 + np.dot(
                    DtD,
                    (
                        np.dot(y, (xk.T).reshape((1, x_dim)))
                        - np.dot(np.diag((y**2).reshape(s_dim)), W1)
                    ),
                )
                W2 = W2 + np.dot(
                    DtD,
                    (np.dot(y, y.T) - np.dot(np.diag((y**2).reshape(s_dim)), W2)),
                )

                for ind in range(s_dim):
                    W2[ind, ind] = 0

                nzerocount = (nzerocount + (y.reshape(s_dim) == 0) * 1.0) * (
                    y.reshape(s_dim) == 0
                )
                if i_sample < ZERO_CHECK_INTERVAL:
                    q = np.argwhere(nzerocount > 50)
                    qq = q[:, 0]
                    for iter3 in range(len(qq)):
                        W1[qq[iter3], :] = -W1[qq[iter3], :]
                        nzerocount[qq[iter3]] = 0

                self.W1 = W1
                self.W2 = W2
                self.Dt = Dt

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                                Wf, A, S, X, False
                            )
                            self.SV_list.append(abs(SGG))

                            SIR_list.append(SINR)
                            SNR_list.append(SNR)

                            self.SNR_list = SNR_list
                            self.SIR_list = SIR_list
                            self.SV_list.append(abs(SGG))

                            if plot_in_jupyter:
                                YforPlot = Y[:, idx[i_sample - 25 : i_sample]].T
                                self.plot_for_debug(
                                    SIR_list,
                                    SNR_list,
                                    P,
                                    debug_iteration_point,
                                    YforPlot,
                                )
                        except Exception as e:
                            print(str(e))
