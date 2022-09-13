###### NUMBA UTILITY FUNCTIONS FOR FASTER IMPLEMENTATION ###################
import numpy as np
from numba import njit


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


@njit(parallel=True)
def snr_jit(S_original, S_noisy):
    N_hat = S_original - S_noisy
    N_P = (N_hat**2).sum(axis=1)
    S_P = (S_original**2).sum(axis=1)
    snr = 10 * np.log10(S_P / N_P)
    return snr

@njit(fastmath=True)
def accumu(lis):
    """Cumulative Sum.

    Same as np.cumsum()
    """
    result = np.zeros_like(lis)
    for i in range(lis.shape[1]):
        result[:, i] = np.sum(lis[:, : i + 1])

    return result


@njit(fastmath=True)
def merge_sort(list_):
    """Sorts a list in ascending order. Returns a new sorted list.

    Divide : Find the midpoint of the list and divide into sublist
    Conquer : Recursively sort the sublists created in previous step
    Combine : Merge the sorted sublists created in previous step

    Takes O(n log n) time.
    """

    def merge(left, right):
        """Merges two lists (arrays), sorting them in the process. Returns a new merged
        list.

        Runs in overall O(n) time
        """

        l = []
        i = 0
        j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                l.append(left[i])
                i += 1
            else:
                l.append(right[j])
                j += 1

        while i < len(left):
            l.append(left[i])
            i += 1

        while j < len(right):
            l.append(right[j])
            j += 1

        return l

    def split(list_):
        """Divide the unsorted list at midpoint into sublists.

        Returns two sublists - left and right

        Takes overall O(log n) time
        """

        mid = len(list_) // 2

        left = list_[:mid]
        right = list_[mid:]

        return left, right

    if len(list_) <= 1:
        return list_

    left_half, right_half = split(list_)
    left = merge_sort(left_half)
    right = merge_sort(right_half)

    return np.array(merge(left, right))