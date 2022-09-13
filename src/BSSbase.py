import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm


class BSSBaseClass:
    def whiten_input(self, X, n_components=None, return_prewhitening_matrix=False):
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

    @staticmethod
    @njit
    def ProjectOntoLInfty(X):
        return X * (X >= -1.0) * (X <= 1.0) + (X > 1.0) * 1.0 - 1.0 * (X < -1.0)

    @staticmethod
    @njit
    def ProjectOntoNNLInfty(X):
        return X * (X >= 0.0) * (X <= 1.0) + (X > 1.0) * 1.0  # -0.0*(X<0.0)

    def ProjectRowstoL1NormBall(self, H):
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

    def ProjectColstoSimplex(self, v, z=1):
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

    def ProjectRowstoBoundaryRectangle(self, H, BoundMin, BoundMax):
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
        self, x, signed_dims, nn_dims, sparse_dims_list, max_number_of_iterations=1
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
                x_projected = self.ProjectRowstoBoundaryRectangle(
                    x_projected.T, BoundMin, BoundMax
                ).T
                x_projected[sparse_dims_list[j], :] = self.ProjectRowstoL1NormBall(
                    x_projected[sparse_dims_list[j], :].T
                ).T
        return x_projected

    def CalculateSIR(self, H, pH, return_db=True):
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

    @staticmethod
    @njit
    def CalculateSINRjit(Out, S, compute_permutation=True):
        def mean_numba(a):
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
                S[indmax[kk], :] - Smean[indmax[kk]],
                S[indmax[kk], :].T - Smean[indmax[kk]],
            )  # (G[kk,indmax[kk]])

        ZZ = GG @ (S - np.reshape(Smean, (r, 1))) + np.reshape(Outmean, (r, 1))
        E = Out - ZZ
        MSE = np.linalg.norm(E) ** 2
        SigPow = np.linalg.norm(ZZ) ** 2
        SINR = SigPow / MSE
        return SINR, SigPow, MSE, G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat**2).sum(axis=1)
        S_P = (S_original**2).sum(axis=1)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    @staticmethod
    @njit(parallel=True)
    def snr_jit(S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat**2).sum(axis=1)
        S_P = (S_original**2).sum(axis=1)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick."""
        return A[..., None] * B[:, None]

    def find_permutation_between_source_and_estimation(self, S, Y):
        """
        S    : Original source matrix
        Y    : Matrix of estimations of sources (after BSS or ICA algorithm)

        return the permutation of the source seperation algorithm
        """
        # perm = np.argmax(np.abs(self.outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
        perm = np.argmax(
            np.abs(self.outer_prod_broadcasting(Y.T, S.T).sum(axis=0))
            / (np.linalg.norm(S, axis=1) * np.linalg.norm(Y, axis=1)),
            axis=0,
        )
        return perm

    def signed_and_permutation_corrected_sources(self, S, Y):
        """_summary_

        Args:
            S (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        perm = self.find_permutation_between_source_and_estimation(S, Y)
        return (np.sign((Y[perm, :] * S).sum(axis=1))[:, np.newaxis]) * Y[perm, :]

    def find_permutation_between_source_and_estimationV2(self, S, Y):
        """
        S    : Original source matrix
        Y    : Matrix of estimations of sources (after BSS or ICA algorithm)

        return the permutation of the source seperation algorithm
        """
        number_of_sources = S.shape[0]
        perm = np.argmax(np.corrcoef(S, Y)[number_of_sources:,:number_of_sources],axis=0)
        return perm

    def signed_and_permutation_corrected_sourcesV2(self, S, Y):
        perm = self.find_permutation_between_source_and_estimationV2(S, Y)
        return (np.sign((Y[perm, :] * S).sum(axis=1))[:, np.newaxis]) * Y[perm, :]