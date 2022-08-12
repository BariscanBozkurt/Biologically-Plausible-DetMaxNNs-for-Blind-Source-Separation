####### BLIND SOURCE SEPARATION UTILITY FUNCTIONS #############
from tokenize import Number
import numpy as np
from scipy.stats import invgamma, chi2, t
from scipy import linalg
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull  
from numpy.linalg import det
from scipy.stats import dirichlet
from scipy import signal
import mne
# from mne.preprocessing import ICA


########## SOME INDEPENDENT COMPONENT ANALYSIS (ICA) FUNCTIONS ######################
def fit_icainfomax(X, NumberofSources = None, ch_types = None, n_subgauss = None, max_iter = 10000, verbose = False):
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
    mneinfo = mne.create_info(NumberofMixtures, 2000, ch_types = ch_types)
    mneobj = mne.io.RawArray(X, mneinfo)
    ica = mne.preprocessing.ICA(n_components = NumberofSources, method = "infomax",
                                fit_params = {"extended": True, "n_subgauss":n_subgauss,"max_iter":max_iter},
                                random_state = 1,verbose = verbose)
    ica.fit(mneobj)
    Se = ica.get_sources(mneobj)
    Y = Se.get_data()
    return Y

def fobi(X, return_unmixing_matrix = False):
    """
    Blind source separation via the FOBI (fourth order blind identification).
    Algorithm is based on the descriptions in the paper 
    "A Normative and Biologically Plausible Algorithm for Independent Component Analysis (Neurips2021)"
    See page 3 and 4.
    """
    n_samples = X.shape[1]
    muX = np.mean(X, 1).reshape(-1,1)
    Cx = (1/n_samples) * (X - muX) @ (X - muX).T
    Cx_square_root = np.linalg.cholesky(Cx)
    H = np.linalg.pinv(Cx_square_root) @ X
    norm_h = (np.sum(np.abs(H)**2,axis=0)**(1./2))
    Z = norm_h * H
    _, _, W = np.linalg.svd((1/n_samples)*Z @ Z.T)
    Y = W @ H
    if return_unmixing_matrix:
        return Y, W @ np.linalg.pinv(Cx_square_root)
    else:
        return Y
       
####### SIGN AND PERMUTATION CORRECTION  FUNCTIONS ####################
def outer_prod_broadcasting(A, B):
    """Broadcasting trick"""
    return A[...,None]*B[:,None]

def find_permutation_between_source_and_estimation(S,Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
    
    return the permutation of the source seperation algorithm
    """
    perm = np.argmax(np.abs(outer_prod_broadcasting(Y.T,S.T).sum(axis = 0))/(np.linalg.norm(S,axis = 1)*np.linalg.norm(Y,axis=1)), axis = 0)
    return perm

def signed_and_permutation_corrected_sources(S,Y):
    perm = find_permutation_between_source_and_estimation(S,Y)
    return (np.sign((Y[perm,:] * S).sum(axis = 1))[:,np.newaxis]) * Y[perm,:]

########### PROJECTION AND NORMALIZATION FUNCTIONS #####################
def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:,i] = ZeroOneNormalizeData(X[:,i])

    return X_normalized

def ProjectOntoLInfty(X):
    return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)

def ProjectRowstoL1NormBall(H):
    Hshape=H.shape
    #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
    lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
    #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

    u=-np.sort(-np.abs(H),axis=1)
    sv=np.cumsum(u,axis=1)
    q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
    rho=np.max(q,axis=1)
    rho=rho.astype(int)
    lindex=np.linspace(1,Hshape[0],Hshape[0])-1
    lindex=lindex.astype(int)
    theta=np.maximum(0,np.reshape((sv[tuple([lindex,rho])]-1)/(rho+1),(Hshape[0],1)))
    ww=np.abs(H)-theta
    H=np.sign(H)*(ww>0)*ww
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

########### SIGNAL TO INTERFERENCE-PLUS-NOISE RATIO FUNCTIONS ##################
def CalculateSIR(H,pH, return_db = True):
    """_summary_

    Args:
        H (_type_): _description_
        pH (_type_): _description_
        return_db (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    G=pH@H
    Gmax=np.diag(np.max(abs(G),axis=1))
    P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
    T=G@P.T
    rankP=np.linalg.matrix_rank(P)
    diagT = np.diag(T)
    # Signal Power
    sigpow = np.linalg.norm(diagT,2)**2
    # Interference Power
    intpow = np.linalg.norm(T,'fro')**2 - sigpow
    SIRV = sigpow/intpow
    # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
    if return_db:
        SIRV = 10*np.log10(sigpow/intpow)

    return SIRV,rankP

def CalculateSINR(Out,S, compute_permutation = True):
    """_summary_

    Args:
        Out (_type_): _description_
        S (_type_): _description_
        compute_permutation (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    Smean = np.mean(S,1)
    Outmean = np.mean(Out,1)
    r=S.shape[0]
    if compute_permutation:
        G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
        indmax=np.argmax(np.abs(G),1)
    else:
        G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
        indmax = np.arange(0,r)
        
    GG=np.zeros((r,r))
    for kk in range(r):
        GG[kk,indmax[kk]]=np.dot(Out[kk,:] - Outmean[kk], S[indmax[kk],:].T - Smean[indmax[kk]])/np.dot(S[indmax[kk],:] - Smean[indmax[kk]], S[indmax[kk],:].T - Smean[indmax[kk]])#(G[kk,indmax[kk]])

    ZZ = GG @ (S-np.reshape(Smean,(r,1))) + np.reshape(Outmean,(r,1))
    E=Out-ZZ
    MSE=np.linalg.norm(E,'fro')**2
    SigPow=np.linalg.norm(ZZ,'fro')**2
    SINR=(SigPow/MSE)
    return SINR,SigPow,MSE,G

############# SYNTHETIC DATA AND LINEAR MIXING FUNCTIONS #################
def generate_correlated_uniform_sources(R, range_ = [-1,1], n_sources = 5, size_sources = 500000):
    """
    R : correlation matrix
    """
    assert R.shape[0] == n_sources, "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)" % (n_sources)
    S = np.random.uniform(range_[0], range_[1], size = (n_sources, size_sources))
    L = np.linalg.cholesky(R)
    S_ = L @ S
    return S_

def generate_correlated_copula_sources(rho = 0.0, df = 4, n_sources = 5, size_sources = 500000, decreasing_correlation = True):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho ** j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size = size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df/s) * Z # chi-square method
    S = t.cdf(X, df).T
    return S

def generate_uniform_points_in_polytope(polytope_vertices, size):
    """"
    polytope_vertices : vertex matrix of shape (n_dim, n_vertices)

    return:
        Samples of shape (n_dim, size)
    """
    polytope_vertices = polytope_vertices.T
    dims = polytope_vertices.shape[-1]
    hull = polytope_vertices[ConvexHull(polytope_vertices).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = size, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = size)).T

def generate_uniform_points_in_simplex(NumberofSources, NumberofSamples, gain = 1):
    S = np.random.exponential(scale=1.0, size = (NumberofSources, NumberofSamples))
    S = gain * (S / np.sum(S, axis = 0))
    return S

def WSM_Mixing_Scenario(S, NumberofMixtures = None, INPUT_STD = None):
    NumberofSources = S.shape[0]
    if INPUT_STD is None:
        INPUT_STD = S.std()
    if NumberofMixtures is None:
        NumberofMixtures = NumberofSources
    A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
    X = A @ S
    for M in range(A.shape[0]):
        stdx = np.std(X[M,:])
        A[M,:] = A[M,:]/stdx * INPUT_STD
    X = A @ S
    return A, X

def generate_synthetic_data_SMICA(seed=101, samples=10000):
    """This function is taken from the original published code of the 
       paper 'A Normative and Biologically Plausible Algorithm for Independent Component Analysis' """
    mix_dim=4
    np.random.seed(seed)
    t = np.linspace(0, samples * 1e-4, samples)
    two_pi = 2 * np.pi
    s0 = np.sign(np.cos(two_pi * 155 * t))
    s1 = np.sin(two_pi * 180 * t)
    s2= signal.sawtooth(2 * np.pi*200  * t)
    s3 = np.random.laplace(0, 1, (samples,))
    S = np.stack([s0, s1, s2,s3])
    A = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
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

    U = np.random.uniform(0,np.sqrt(48/5),(s_dim,samples)) # independent non-negative uniform source RVs with variance 1
    B = np.random.binomial(1, .5, (s_dim,samples)) # binomial RVs to sparsify the source
    S = U*B # sources
    
    A = np.random.randn(x_dim,s_dim) # random mixing matrix

    # Generate mixtures
    
    X = A @ S

    return S, X, A
