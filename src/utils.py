import itertools
import logging
import math
import os
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pylab as pl
import pypoman
import scipy
from IPython.display import Latex, Math, clear_output, display
from matplotlib.pyplot import draw, plot, show
from mne.preprocessing import ICA
from numba import jit, njit
from numpy.linalg import det
from scipy import linalg, signal
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import chi2, dirichlet, invgamma, t
from tqdm import tqdm

############################ UTILITY FUNCTIONS ######################################

# class Timer:
#     # It requires import time whereas in this script we have "from time import time"
#     # A simple timer class for performance profiling
#     # Taken from https://github.com/flatironinstitute/online_psp/blob/master/online_psp/online_psp_simulations.py
#     """
#     Usage:
#     with Timer() as t:
#         DO SOMETHING HERE
#     print('Above (DO SOMETHING HERE) took %f sec.' % (t.interval))
#     """
#     def __enter__(self):
#         self.start = time.perf_counter()
#         return self

#     def __exit__(self, *args):
#         self.end = time.perf_counter()
#         self.interval = self.end - self.start


def fobi(X, return_unmixing_matrix=False):
    """Blind source separation via the FOBI (fourth order blind identification).

    Algorithm is based on the descriptions in the paper "A Normative and
    Biologically Plausible Algorithm for Independent Component Analysis
    (Neurips2021)" See page 3 and 4.
    """
    n_samples = X.shape[1]
    muX = np.mean(X, 1).reshape(-1, 1)
    display_matrix((1 / X.shape[1]) * (X - muX) @ (X - muX).T)
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


def whiten_signal(X, mean_normalize=True, type_=3):
    """
    Input : X  ---> Input signal to be whitened

    type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.

    Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
    """
    if mean_normalize:
        X = X - np.mean(X, axis=0, keepdims=True)

    cov = np.cov(X.T)

    if type_ == 3:  # Whitening using singular value decomposition
        U, S, V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        W_pre = np.dot(U, np.dot(d, U.T))

    else:  # Whitening using eigenvalue decomposition
        d, S = np.linalg.eigh(cov)
        D = np.diag(d)

        D_sqrt = np.sqrt(D * (D > 0))

        if type_ == 1:  # Type defines how you want W_pre matrix to be
            W_pre = np.linalg.pinv(S @ D_sqrt)
        elif type_ == 2:
            W_pre = np.linalg.pinv(S @ D_sqrt @ S.T)

    X_white = (W_pre @ X.T).T

    return X_white, W_pre


def whiten_input(X, n_components=None, return_prewhitening_matrix=False):
    """
    X.shape[0] = Number of sources
    X.shape[1] = Number of samples for each signal
    """
    x_dim = X.shape[0]
    if n_components is None:
        n_components = x_dim
    s_dim = n_components

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
    if return_prewhitening_matrix:
        return H, Wpre
    else:
        return H


def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:, i] = ZeroOneNormalizeData(X[:, i])

    return X_normalized


def ProjectOntoLInfty(X):
    return X * (X >= -1.0) * (X <= 1.0) + (X > 1.0) * 1.0 - 1.0 * (X < -1.0)


def Subplot_gray_images(I, image_shape=[512, 512], height=15, width=15, title=""):
    n_images = I.shape[1]
    fig, ax = plt.subplots(1, n_images)
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:, i].reshape(image_shape[0], image_shape[1]), cmap="gray")

    plt.show()


def subplot_1D_signals(
    X, title="", title_fontsize=20, figsize=(10, 5), linewidth=1, colorcode="#050C12"
):
    """Plot the 1D signals (each row from the given matrix)"""
    n = X.shape[0]  # Number of signals

    fig, ax = plt.subplots(n, 1, figsize=figsize)

    for i in range(n):
        ax[i].plot(X[i, :], linewidth=linewidth, color=colorcode)
        ax[i].grid()

    plt.suptitle(title, fontsize=title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()


def plot_convergence_plot(
    metric,
    xlabel="",
    ylabel="",
    title="",
    figsize=(12, 8),
    fontsize=15,
    linewidth=3,
    colorcode="#050C12",
):

    plt.figure(figsize=figsize)
    plt.plot(metric, linewidth=linewidth, color=colorcode)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.draw()


def outer_prod_broadcasting(A, B):
    """Broadcasting trick."""
    return A[..., None] * B[:, None]


def find_permutation_between_source_and_estimation(S, Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)

    return the permutation of the source seperation algorithm
    """

    # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    # perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    # perm = np.argmax(np.abs(outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
    perm = np.argmax(
        np.abs(outer_prod_broadcasting(Y.T, S.T).sum(axis=0))
        / (np.linalg.norm(S, axis=1) * np.linalg.norm(Y, axis=1)),
        axis=0,
    )
    return perm


def signed_and_permutation_corrected_sources(S, Y):
    perm = find_permutation_between_source_and_estimation(S, Y)
    return (np.sign((Y[perm, :] * S).sum(axis=1))[:, np.newaxis]) * Y[perm, :]
    # return np.sign((Y[perm,:] * S).sum(axis = 0)) * Y[perm,:]


def ProjectRowstoL1NormBall(H):
    Hshape = H.shape
    # lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
    lr = np.tile(
        np.reshape((1 / np.linspace(1, Hshape[1], Hshape[1])), (1, Hshape[1])),
        (Hshape[0], 1),
    )
    # Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

    u = -np.sort(-np.abs(H), axis=1)
    sv = np.cumsum(u, axis=1)
    q = np.where(
        u > ((sv - 1) * lr),
        np.tile(
            np.reshape((np.linspace(1, Hshape[1], Hshape[1]) - 1), (1, Hshape[1])),
            (Hshape[0], 1),
        ),
        np.zeros((Hshape[0], Hshape[1])),
    )
    rho = np.max(q, axis=1)
    rho = rho.astype(int)
    lindex = np.linspace(1, Hshape[0], Hshape[0]) - 1
    lindex = lindex.astype(int)
    theta = np.maximum(
        0, np.reshape((sv[tuple([lindex, rho])] - 1) / (rho + 1), (Hshape[0], 1))
    )
    ww = np.abs(H) - theta
    H = np.sign(H) * (ww > 0) * ww
    return H


def display_matrix(array):
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))


# Calculate SIR Function
def CalculateSIR(H, pH, return_db=True):
    """_summary_

    Args:
        H (_type_): _description_
        pH (_type_): _description_
        return_db (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    G = pH @ H
    Gmax = np.diag(np.max(abs(G), axis=1))
    P = 1.0 * ((np.linalg.inv(Gmax) @ np.abs(G)) > 0.99)
    T = G @ P.T
    rankP = np.linalg.matrix_rank(P)
    diagT = np.diag(T)
    # Signal Power
    sigpow = np.linalg.norm(diagT, 2) ** 2
    # Interference Power
    intpow = np.linalg.norm(T, "fro") ** 2 - sigpow
    SIRV = sigpow / intpow
    # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
    if return_db:
        SIRV = 10 * np.log10(sigpow / intpow)

    return SIRV, rankP


def CalculateSINR(Out, S, compute_permutation=True):
    """_summary_

    Args:
        Out (_type_): _description_
        S (_type_): _description_
        compute_permutation (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    Smean = np.mean(S, 1)
    Outmean = np.mean(Out, 1)
    r = S.shape[0]
    if compute_permutation:
        G = np.dot(
            Out - np.reshape(np.mean(Out, 1), (r, 1)),
            np.linalg.pinv(S - np.reshape(np.mean(S, 1), (r, 1))),
        )
        indmax = np.argmax(np.abs(G), 1)
    else:
        G = np.dot(
            Out - np.reshape(np.mean(Out, 1), (r, 1)),
            np.linalg.pinv(S - np.reshape(np.mean(S, 1), (r, 1))),
        )
        indmax = np.arange(0, r)

    GG = np.zeros((r, r))
    for kk in range(r):
        GG[kk, indmax[kk]] = np.dot(
            Out[kk, :] - Outmean[kk], S[indmax[kk], :].T - Smean[indmax[kk]]
        ) / np.dot(
            S[indmax[kk], :] - Smean[indmax[kk]], S[indmax[kk], :].T - Smean[indmax[kk]]
        )  # (G[kk,indmax[kk]])

    ZZ = GG @ (S - np.reshape(Smean, (r, 1))) + np.reshape(Outmean, (r, 1))
    E = Out - ZZ
    MSE = np.linalg.norm(E, "fro") ** 2
    SigPow = np.linalg.norm(ZZ, "fro") ** 2
    SINR = SigPow / MSE
    return SINR, SigPow, MSE, G


@njit
def CalculateSINRjit(Out, S, compute_permutation=True):
    """_summary_

    Args:
        Out (_type_): _description_
        S (_type_): _description_
        compute_permutation (bool, optional): _description_. Defaults to True.
    """

    def mean_numba(a):
        """_summary_

        Args:
            a (_type_): _description_

        Returns:
            _type_: _description_
        """
        res = []
        for i in range(a.shape[0]):
            res.append(a[i, :].mean())

        return np.array(res)

    r = S.shape[0]
    Smean = mean_numba(S)
    Outmean = mean_numba(Out)
    if compute_permutation:
        G = np.dot(
            Out - np.reshape(Outmean, (r, 1)),
            np.linalg.pinv(S - np.reshape(Smean, (r, 1))),
        )
        # G = np.linalg.lstsq((S-np.reshape(Smean,(r,1))).T, (Out-np.reshape(Outmean,(r,1))).T)[0]
        indmax = np.abs(G).argmax(1).astype(np.int64)
    else:
        G = np.dot(
            Out - np.reshape(Outmean, (r, 1)),
            np.linalg.pinv(S - np.reshape(Smean, (r, 1))),
        )
        # G = np.linalg.lstsq((S-np.reshape(Smean,(r,1))).T, (Out-np.reshape(Outmean,(r,1))).T)[0]
        indmax = np.arange(0, r)

    GG = np.zeros((r, r))
    for kk in range(r):
        GG[kk, indmax[kk]] = np.dot(
            Out[kk, :] - Outmean[kk], S[indmax[kk], :].T - Smean[indmax[kk]]
        ) / np.dot(
            S[indmax[kk], :] - Smean[indmax[kk]], S[indmax[kk], :].T - Smean[indmax[kk]]
        )  # (G[kk,indmax[kk]])

    ZZ = GG @ (S - np.reshape(Smean, (r, 1))) + np.reshape(Outmean, (r, 1))
    E = Out - ZZ
    MSE = np.linalg.norm(E) ** 2
    SigPow = np.linalg.norm(ZZ) ** 2
    SINR = SigPow / MSE
    return SINR, SigPow, MSE, G


def psnr(img1, img2, pixel_max=1):
    """Return peak-signal-to-noise-ratio between given two images (or two signals)

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_
        pixel_max (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(pixel_max / np.sqrt(mse))


def snr(S_original, S_noisy):
    """_summary_

    Args:
        S_original (_type_): _description_
        S_noisy (_type_): _description_

    Returns:
        _type_: _description_
    """
    N_hat = S_original - S_noisy
    N_P = (N_hat**2).sum(axis=1)
    S_P = (S_original**2).sum(axis=1)
    snr = 10 * np.log10(S_P / N_P)
    return snr


@njit(parallel=True)
def snr_jit(S_original, S_noisy):
    N_hat = S_original - S_noisy
    N_P = (N_hat**2).sum(axis=1)
    S_P = (S_original**2).sum(axis=1)
    snr = 10 * np.log10(S_P / N_P)
    return snr


def addWGN(signal, SNR, return_noise=False, print_resulting_SNR=False):
    """
    Adding white Gaussian Noise to the input signal
    signal              : Input signal, numpy array of shape (number of sources, number of samples)
                          If your signal is a 1D numpy array of shape (number of samples, ), then reshape it
                          by signal.reshape(1,-1) before giving it as input to this function
    SNR                 : Desired input signal to noise ratio
    print_resulting_SNR : If you want to print the numerically calculated SNR, pass it as True

    Returns
    ============================
    signal_noisy        : Output signal which is the sum of input signal and additive noise
    noise               : Returns the added noise
    """
    sigpow = np.mean(signal**2, axis=1)
    noisepow = 10 ** (-SNR / 10) * sigpow
    noise = np.sqrt(noisepow)[:, np.newaxis] * np.random.randn(
        signal.shape[0], signal.shape[1]
    )
    signal_noisy = signal + noise
    if print_resulting_SNR:
        SNRinp = 10 * np.log10(
            np.sum(np.mean(signal**2, axis=1)) / np.sum(np.mean(noise**2, axis=1))
        )
        print(f"Input SNR is : {SNRinp}")
    if return_noise:
        return signal_noisy, noise
    else:
        return signal_noisy


def WSM_Mixing_Scenario(S, NumberofMixtures=None, INPUT_STD=None):
    NumberofSources = S.shape[0]
    if INPUT_STD is None:
        INPUT_STD = S.std()
    if NumberofMixtures is None:
        NumberofMixtures = NumberofSources
    A = np.random.standard_normal(size=(NumberofMixtures, NumberofSources))
    X = A @ S
    for M in range(A.shape[0]):
        stdx = np.std(X[M, :])
        A[M, :] = A[M, :] / stdx * INPUT_STD
    X = A @ S
    return A, X


def generate_synthetic_data_SMICA(seed=101, samples=10000):
    """This function is taken from the original published code of the paper 'A Normative
    and Biologically Plausible Algorithm for Independent Component Analysis'."""
    mix_dim = 4
    np.random.seed(seed)
    t = np.linspace(0, samples * 1e-4, samples)
    two_pi = 2 * np.pi
    s0 = np.sign(np.cos(two_pi * 155 * t))
    s1 = np.sin(two_pi * 180 * t)
    s2 = signal.sawtooth(2 * np.pi * 200 * t)
    s3 = np.random.laplace(0, 1, (samples,))
    S = np.stack([s0, s1, s2, s3])
    A = np.random.uniform(-0.5, 0.5, (mix_dim, mix_dim))
    X = np.dot(A, S)
    return S, X, A


def synthetic_data(s_dim, x_dim, samples):
    """
    Parameters:
    ====================
    s_dim   -- The dimension of sources
    x_dim   -- The dimension of mixtures
    samples -- The number of samples
    Output:
    ====================
    S       -- The source data matrix
    X       -- The mixture data matrix
    A       -- Mixing Matrix
    """

    # Generate sparse random samples

    U = np.random.uniform(
        0, np.sqrt(48 / 5), (s_dim, samples)
    )  # independent non-negative uniform source RVs with variance 1
    B = np.random.binomial(
        1, 0.5, (s_dim, samples)
    )  # binomial RVs to sparsify the source
    S = U * B  # sources

    A = np.random.randn(x_dim, s_dim)  # random mixing matrix

    # Generate mixtures

    X = A @ S

    return S, X, A


def generate_correlated_uniform_sources(
    R, range_=[-1, 1], n_sources=5, size_sources=500000
):
    """
    R : correlation matrix
    """
    assert R.shape[0] == n_sources, (
        "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)"
        % (n_sources)
    )
    S = np.random.uniform(range_[0], range_[1], size=(n_sources, size_sources))
    L = np.linalg.cholesky(R)
    S_ = L @ S
    return S_


def generate_correlated_copula_sources(
    rho=0.0, df=4, n_sources=5, size_sources=500000, decreasing_correlation=True
):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho**j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = (
            np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho
        )

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size=size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df / s) * Z  # chi-square method
    S = t.cdf(X, df).T
    return S


def generate_uniform_points_in_polytope(polytope_vertices, size):
    """ "
    polytope_vertices : vertex matrix of shape (n_dim, n_vertices)

    return:
        Samples of shape (n_dim, size)
    """
    polytope_vertices = polytope_vertices.T
    dims = polytope_vertices.shape[-1]
    hull = polytope_vertices[ConvexHull(polytope_vertices).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    sample = np.random.choice(len(vols), size=size, p=vols / vols.sum())

    return np.einsum(
        "ijk, ij -> ik", deln[sample], dirichlet.rvs([1] * (dims + 1), size=size)
    ).T


def generate_practical_polytope(
    dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list
):
    A = []
    b = []
    for j in antisparse_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(1)

    for j in nonnegative_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(0)

    for relative_sparse_dims in relative_sparse_dims_list:
        row = np.zeros(dim)
        pm_one = [[1, -1] for _ in range(relative_sparse_dims.shape[0])]
        for i in itertools.product(*pm_one):
            row_copy = row.copy()
            row_copy[relative_sparse_dims] = i
            A.append(list(row_copy))
            b.append(1)
    A = np.array(A)
    b = np.array(b)
    vertices = pypoman.compute_polytope_vertices(A, b)
    V = np.array([list(v) for v in vertices]).T
    return (A, b), V
