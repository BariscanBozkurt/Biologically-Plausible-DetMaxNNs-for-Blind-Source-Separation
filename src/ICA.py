import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
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

def fit_icainfomax(
    X,
    NumberofSources=None,
    ch_types=None,
    n_subgauss=None,
    max_iter=10000,
    verbose=False,
):
    """
    X : Mixture Signals, X.shape = (NumberofMixtures, NumberofSamples)

    for more information, visit:
    https://mne.tools/stable/generated/mne.preprocessing.ICA.html

    USAGE:
    Y = fit_icainfomax(X = X, NumberofSources = 3)
    IF GROUND TRUTH IS AVAILABLE:
    Y_ = signed_and_permutation_corrected_sources(S.T, Y.T).T
    """
    NumberofMixtures = X.shape[0]
    if NumberofSources is None:
        NumberofSources = NumberofMixtures
    if ch_types is None:
        ch_types = ["eeg"] * NumberofMixtures
    if n_subgauss is None:
        n_subgauss = NumberofSources
    mneinfo = mne.create_info(NumberofMixtures, 2000, ch_types=ch_types)
    mneobj = mne.io.RawArray(X, mneinfo)
    ica = mne.preprocessing.ICA(
        n_components=NumberofSources,
        method="infomax",
        fit_params={"extended": True, "n_subgauss": n_subgauss, "max_iter": max_iter},
        random_state=1,
        verbose=verbose,
    )
    ica.fit(mneobj)
    Se = ica.get_sources(mneobj)
    Y = Se.get_data()
    return Y

def fobi(X, return_unmixing_matrix=False):
    """Blind source separation via the FOBI (fourth order blind identification).

    Algorithm is based on the descriptions in the paper "A Normative and
    Biologically Plausible Algorithm for Independent Component Analysis
    (Neurips2021)" See page 3 and 4.
    """
    n_samples = X.shape[1]
    muX = np.mean(X, 1).reshape(-1, 1)
    Cx = (1 / n_samples) * (X - muX) @ (X - muX).T
    Cx_square_root = np.linalg.cholesky(Cx)
    H = np.linalg.pinv(Cx_square_root) @ X
    norm_h = np.sum(np.abs(H) ** 2, axis=0) ** (1.0 / 2)
    Z = norm_h * H
    _, _, W = np.linalg.svd((1 / n_samples) * Z @ Z.T)
    Y = W @ H
    if return_unmixing_matrix:
        return Y, W @ np.linalg.pinv(Cx_square_root)
    else:
        return Y

class FastICA(BSSBaseClass):
    def __init__(self, s_dim, x_dim) -> None:
        super().__init__()
        W = np.zeros((s_dim, s_dim), dtype=np.float32)

        self.s_dim = s_dim
        self.x_dim = x_dim

        self.Wpre = np.eye(s_dim, x_dim)
        self.W = W

    def compute_overall_mapping(self, return_mapping=False):
        self.Woverall = self.W @ self.Wpre
        if return_mapping:
            return self.Woverall
        else:
            return None

    def predict(self, X):
        W = self.compute_overall_mapping(return_mapping=True)
        Xcentered = X - X.mean(axis=1, keepdims=True)
        return W @ Xcentered

    def dtanh(self, x):
        return 1 - np.tanh(x) ** 2

    def update_weights(self, w, X):
        w_new = (X * np.tanh(np.dot(w.T, X))).mean(axis=1) - self.dtanh(
            np.dot(w.T, X)
        ).mean() * w
        w_new /= np.sqrt((w_new**2).sum())
        return w_new

    def fit_transform(self, X, n_epochs=1, n_iterations=1000, convergence_tol=1e-6):
        W = self.W
        s_dim = self.s_dim

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        Xwhite = X - X.mean(axis=1, keepdims=True)
        Xwhite, Wpre = whiten_input(X, s_dim, True)
        self.Wpre = Wpre
        for _ in tqdm(range(n_epochs)):
            for i in range(s_dim):
                w = np.random.rand(s_dim)
                for k in range(n_iterations):
                    w_new = self.update_weights(w, Xwhite)
                    if i >= 1:
                        w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

                    diff = np.abs(np.abs((w * w_new).sum()) - 1)
                    w = w_new
                    if diff < convergence_tol:
                        break

                    W[i, :] = w

        self.W = W
        Y = W @ Xwhite

        return Y
