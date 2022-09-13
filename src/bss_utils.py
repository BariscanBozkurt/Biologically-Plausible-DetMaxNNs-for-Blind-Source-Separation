####### BLIND SOURCE SEPARATION UTILITY FUNCTIONS #############
from tokenize import Number

import numpy as np
from numpy.linalg import det
from scipy import linalg, signal
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import chi2, dirichlet, invgamma, t

from numba_utils import CalculateSINRjit, snr_jit


####### SIGN AND PERMUTATION CORRECTION  FUNCTIONS ####################
def outer_prod_broadcasting(A, B):
    """Broadcasting trick."""
    return A[..., None] * B[:, None]


def find_permutation_between_source_and_estimation(S, Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)

    return the permutation of the source seperation algorithm
    """
    perm = np.argmax(
        np.abs(outer_prod_broadcasting(Y.T, S.T).sum(axis=0))
        / (np.linalg.norm(S, axis=1) * np.linalg.norm(Y, axis=1)),
        axis=0,
    )
    return perm

def find_permutation_between_source_and_estimationV2(S, Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)

    return the permutation of the source seperation algorithm
    """
    number_of_sources = S.shape[0]
    perm = np.argmax(np.corrcoef(S, Y)[number_of_sources:,:number_of_sources],axis=0)
    return perm

def signed_and_permutation_corrected_sources(S, Y):
    perm = find_permutation_between_source_and_estimation(S, Y)
    return (np.sign((Y[perm, :] * S).sum(axis=1))[:, np.newaxis]) * Y[perm, :]

def signed_and_permutation_corrected_sourcesV2(S, Y):
    perm = find_permutation_between_source_and_estimationV2(S, Y)
    return (np.sign((Y[perm, :] * S).sum(axis=1))[:, np.newaxis]) * Y[perm, :]

########### PROJECTION AND NORMALIZATION FUNCTIONS #####################
def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:, i] = ZeroOneNormalizeData(X[:, i])

    return X_normalized


def ProjectOntoLInfty(X):
    return X * (X >= -1.0) * (X <= 1.0) + (X > 1.0) * 1.0 - 1.0 * (X < -1.0)


def ProjectRowstoL1NormBall(H):
    Hshape = H.shape
    lr = np.tile(
        np.reshape((1 / np.linspace(1, Hshape[1], Hshape[1])), (1, Hshape[1])),
        (Hshape[0], 1),
    )
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


def ProjectColstoSimplex(v, z=1):
    """v array of shape (n_features, n_samples)."""
    p, n = v.shape
    u = np.sort(v, axis=0)[::-1, ...]
    pi = np.cumsum(u, axis=0) - z
    ind = (np.arange(p) + 1).reshape(-1, 1)
    mask = (u - pi / ind) > 0
    rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
    theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def ProjectRowstoBoundaryRectangle(H, BoundMin, BoundMax):
    Hshape = H.shape
    a0 = 1 - 2 * (np.sum(H, axis=0) < 0) * (BoundMin == 0)
    AA0 = np.diag(np.reshape(a0, (-1,)))
    H = np.dot(H, AA0)
    BoundMaxlist = np.dot(np.ones((Hshape[0], 1)), BoundMax)
    BoundMinlist = np.dot(np.ones((Hshape[0], 1)), BoundMin)
    CheckMin = 1.0 * (H > BoundMinlist)
    a = 1 - 2.0 * (np.sum(CheckMin, axis=0) == 0) * (BoundMin == 0)
    AA = np.diag(np.reshape(a, (-1,)))
    H = np.dot(H, AA)
    CheckMax = 1.0 * (H < BoundMaxlist)
    CheckMin = 1.0 * (H > BoundMinlist)
    H = (
        H * CheckMax * CheckMin
        + (1 - CheckMin) * BoundMinlist
        + (1 - CheckMax) * BoundMaxlist
    )
    return H


def ProjectColumnsOntoPracticalPolytope(
    x, signed_dims, nn_dims, sparse_dims_list, max_number_of_iterations=1
):
    dim = len(signed_dims) + len(nn_dims)
    BoundMax = np.ones((dim, 1)).T
    BoundMin = -np.ones((dim, 1)).T
    BoundMin[:, nn_dims] = 0
    x_projected = x.copy()
    x_projected[signed_dims, :] = np.clip(x_projected[signed_dims, :], -1, 1)
    x_projected[nn_dims, :] = np.clip(x_projected[nn_dims, :], 0, 1)
    for kk in range(max_number_of_iterations):
        for j in range(len(sparse_dims_list)):
            x_projected = ProjectRowstoBoundaryRectangle(
                x_projected.T, BoundMin, BoundMax
            ).T
            x_projected[sparse_dims_list[j], :] = ProjectRowstoL1NormBall(
                x_projected[sparse_dims_list[j], :].T
            ).T
    return x_projected


########### SIGNAL TO INTERFERENCE-PLUS-NOISE RATIO FUNCTIONS ##################
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
        Out (numpy array): Estimated source vectors by BSS algorithm (of shape (NumberofSources, NumberofSamples))
        S (numpy array): Ground Truth Source Vectors (of shape (NumberofSources, NumberofSamples))
        compute_permutation (bool, optional): _description_. Defaults to True.

    Returns:
        SINR (float): Signal-to-Interference-plus-Noise-Ratio
        SigPow
        MSE
        G
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

####### BSS Algorithm Evaluation #################################
def evaluate_bss(W, Y, A, S, mean_normalize_estimations=False):
    """_summary_

    Args:
        W (numpy array): Trained separator matrix (from a BSS algorithm)
        Y (numpy array): Estimated source vectors by BSS algorithm (of shape (NumberofSources, NumberofSamples))
        A (numpy array): Ground Truth Mixing Matrix
        S (numpy array): Ground Truth Source Vectors (if it is given as mean centered, then pass mean_normalize_estimations = True)
        mean_normalize_estimations (bool, optional): Whether to mean center the source estimation matrix Y or not. Defaults to False.

    Returns:
        SINR (float): Signal-to-Interference-plus-Noise-Ratio
        SNR (numpy array): Signal-to-Noise-Ratio for each source (rows of S are the ground truth source vectors)

    """
    s_dim = S.shape[0]
    if mean_normalize_estimations:
        Y = Y - Y.mean(axis=1, keepdims=True)
    Y_ = signed_and_permutation_corrected_sources(S, Y)
    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
    Y_ = coef_ * Y_

    SINR = 10 * np.log10(CalculateSINRjit(Y_, S, False)[0])
    SNR = snr_jit(S, Y_)

    T = W @ A
    Tabs = np.abs(T)
    P = np.zeros((s_dim, s_dim))

    for SourceIndex in range(s_dim):
        Tmax = np.max(Tabs[SourceIndex, :])
        Tabs[SourceIndex, :] = Tabs[SourceIndex, :] / Tmax
        P[SourceIndex, :] = Tabs[SourceIndex, :] > 0.999

    GG = P.T @ T
    _, SGG, _ = np.linalg.svd(GG)  # SGG is the singular values of overall matrix Wf @ A

    return SINR, SNR, SGG, Y_, P

def evaluate_bssV2(W, Y, A, S, mean_normalize_estimations=False):
    """_summary_

    Args:
        W (numpy array): Trained separator matrix (from a BSS algorithm)
        Y (numpy array): Estimated source vectors by BSS algorithm (of shape (NumberofSources, NumberofSamples))
        A (numpy array): Ground Truth Mixing Matrix
        S (numpy array): Ground Truth Source Vectors (if it is given as mean centered, then pass mean_normalize_estimations = True)
        mean_normalize_estimations (bool, optional): Whether to mean center the source estimation matrix Y or not. Defaults to False.

    Returns:
        SINR (float): Signal-to-Interference-plus-Noise-Ratio
        SNR (numpy array): Signal-to-Noise-Ratio for each source (rows of S are the ground truth source vectors)

    """
    s_dim = S.shape[0]
    if mean_normalize_estimations:
        Y = Y - Y.mean(axis=1, keepdims=True)
    Y_ = signed_and_permutation_corrected_sourcesV2(S, Y)
    coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
    Y_ = coef_ * Y_

    SINR = 10 * np.log10(CalculateSINRjit(Y_, S, False)[0])
    SNR = snr_jit(S, Y_)

    T = W @ A
    Tabs = np.abs(T)
    P = np.zeros((s_dim, s_dim))

    for SourceIndex in range(s_dim):
        Tmax = np.max(Tabs[SourceIndex, :])
        Tabs[SourceIndex, :] = Tabs[SourceIndex, :] / Tmax
        P[SourceIndex, :] = Tabs[SourceIndex, :] > 0.999

    GG = P.T @ T
    _, SGG, _ = np.linalg.svd(GG)  # SGG is the singular values of overall matrix Wf @ A

    return SINR, SNR, SGG, Y_, P
############# SYNTHETIC DATA AND LINEAR MIXING FUNCTIONS #################
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

def generate_uniform_points_in_simplex(NumberofSources, NumberofSamples, gain=1):
    S = np.random.exponential(scale=1.0, size=(NumberofSources, NumberofSamples))
    S = gain * (S / np.sum(S, axis=0))
    return S


def BSS_Gaussian_Mixing_Scenario(
    S, NumberofMixtures=None, gaussian_mean=0, gaussian_std=1
):
    NumberofSources = S.shape[0]
    if NumberofMixtures is None:
        NumberofMixtures = NumberofSources
    # A = np.random.normal(gaussian_mean, gaussian_std, (NumberofMixtures,NumberofSources))
    A = (
        gaussian_std * np.random.randn(NumberofMixtures, NumberofSources)
        + gaussian_mean
    )
    X = np.dot(A, S)
    return A, X


def BSS_Uniform_Mixing_Scenario(
    S, NumberofMixtures=None, uniform_low=0, uniform_high=1
):
    NumberofSources = S.shape[0]
    if NumberofMixtures is None:
        NumberofMixtures = NumberofSources
    A = np.random.uniform(
        uniform_low, uniform_high, (NumberofMixtures, NumberofSources)
    )
    X = np.dot(A, S)
    return A, X


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
    This function is taken from the original published code of the
    paper 'A Normative and Biologically Plausible Algorithm for Independent Component Analysis'
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


# def ProjectColumnsOntoPracticalPolytope(x, signed_dims, nn_dims, sparse_dims_list, max_number_of_iterations = 5):
#     ### THIS PROJECTION IS NOT EXACT, NEED MORE DEBUGGING !!!!!
#     """Projection onto a practical polytope in the form of
#        $\left\{\mathbf{s}\in \mathbb{R}^n\ \middle\vert  s_i\in[-1,1] \, \forall i\in \mathcal{I}_s,\, s_i\in[0,1] \,
#        \forall i\in \mathcal{I}_+, \, \left\|\vs_{\mathcal{J}_k}\right\|_1\le 1, \, \mathcal{J}_k\subseteq \mathbb{Z}_n, \,
#        k\in\mathbb{Z}_L  \right\}$

#     Args:
#         x (numpy array): A point to be projected on the polytope
#         antisparse_dims (numpy array): Antisparse dimension indices
#         nonnegative_dims (numpy array): Nonnegative antisparse dimension indices
#         sparse_dims_list (list of numpy arrays): Collection of mutually sparse dimension indices
#         max_number_of_iterations (int, optional): Maximum number of iterations for alternating projections. Defaults to 1.

#     Returns:
#         _type_: _description_
#     """
#     x_projected = x.copy()
#     x_projected[signed_dims, :] = np.clip(x_projected[signed_dims, :], -1, 1)
#     x_projected[nn_dims, :] = np.clip(x_projected[nn_dims, :], 0, 1)
#     for kk in range(max_number_of_iterations):
#         for j in range(len(sparse_dims_list)):
#             x_projected[signed_dims, :] = np.clip(x_projected[signed_dims, :], -1, 1)
#             x_projected[nn_dims, :] = np.clip(x_projected[nn_dims, :], 0, 1)
#             x_projected[sparse_dims_list[j], :] = ProjectRowstoL1NormBall(x[sparse_dims_list[j], :].T).T
#     return x_projected

# def ProjectOntoPracticalPolytope(x, signed_dims, nn_dims, sparse_dims_list, inequalities = (None, None), max_number_of_iterations = 1, tolerance = 1e-7):
#     x_projected = x.copy()
#     if (inequalities[0] is not None) & (inequalities[1] is not None):
#         A = inequalities[0]
#         b = inequalities[1].reshape(-1,1)
#     else:
#         tolerance = -100

#     x_projected[signed_dims] = np.clip(x_projected[signed_dims], -1, 1)
#     x_projected[nn_dims] = np.clip(x_projected[nn_dims], 0, 1)

#     for kk in range(max_number_of_iterations):
#         for j in range(len(sparse_dims_list)):
#             x_projected[signed_dims] = np.clip(x_projected[signed_dims], -1, 1)
#             x_projected[nn_dims] = np.clip(x_projected[nn_dims], 0, 1)
#             x_projected[sparse_dims_list[j]] = ProjectRowstoL1NormBall(x[sparse_dims_list[j]].T).T
# #             x_projected[signed_dims] = np.clip(x_projected[signed_dims], -1, 1)
# #             x_projected[nn_dims] = np.clip(x_projected[nn_dims], 0, 1)
#         print(np.linalg.norm(np.clip(A @ x_projected - b[:,np.newaxis],0, 1e10)))
#         if np.linalg.norm(np.clip(A @ x_projected - b[:,np.newaxis],0, 1e10)) < tolerance:
#             break
#     return x_projected
