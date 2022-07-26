#### DIGITAL SIGNAL PROCESSING UTILITY FUNCTIONS #######################
import numpy as np
from numba import njit

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

def psnr(img1, img2, pixel_max=1):
    """Return peak-signal-to-noise-ratio between given two images (or two signals)

    Args:
        img1 (numpy array): Reference image
        img2 (numpy array): Predicted (or noisy) image
        pixel_max (int, optional): Maximum pixel value (if your image pixel values are between -1 and 1, pass pixel_max = 2). 
                                   Defaults to 1.

    Returns:
        float: psnr between given two images
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(pixel_max / np.sqrt(mse))

def snr(S_original, S_noisy):
    """Signal to noise ratio (measured for each row)

    Args:
        S_original (numpy array): Ground truth array of signals
        S_noisy (numpy array): Noisy version (or prediction) of S_original

    Returns:
        numpy array: Array containing the SNR values between row
    """
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
        print("Input SNR is : {}".format(SNRinp))
    if return_noise:
        return signal_noisy, noise
    else:
        return signal_noisy

@njit 
def map_estimates_to_symbols(Y, symbols):
    """Mapping a given elements of array Y to the closest symbol in the list of symbols

    Args:
        Y (numpy array): Input array whose elements will be mapped to symbols
        symbols (numpy array): Array containing the alphabet (or symbols)

    Returns:
        numpy array: Mapping each element of Y to closest element in the symbols. Its shape is the same with input Y.

    Example: For 4 PAM example, you can use as follows
    >>> Y = 2 * (np.random.randint(0,4,(NumberofSources, N))) - 3
    >>> Y, NoisePart = addWGN(Y, SNR = 30, return_noise = True)
    >>> Y_pred = map_estimates_to_symbols(Y, np.array([-3, -1, 1, 3]))
    """
    Ysymbols = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            idx = np.argmin(np.abs(Y[i,j] - symbols))
            Ysymbols[i,j] = symbols[idx]
    return Ysymbols

def SER(S, Y):
    """Symbol Error Rate

    Args:
        S (numpy array): Numpy array containing the ground truth symbols
        Y (numpy array): Predicted symbols

    Returns:
        float: Error rate between 0 and 1.
    """
    return np.sum((S - Y) != 0) / (S.size)