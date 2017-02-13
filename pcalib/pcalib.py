"""Sample library demonstrates PCA algorithm using by two approaches:
SVD and eigenvector."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def __calc_pc__(eig_val: np.array, pca_energy: float) -> np.array:
    """Helper function for calculating # of principal components.
    @:arg eig_val - input vector of eig values of N x 1 size (N features per sample).
    @:arg pca_energy - PCA energy, from 0.0 to 1.0.
    @:return # of principal components."""

    denom = sum(eig_val)
    num, k = 0, 0
    for k in range(eig_val.shape[0]):
        num += eig_val[k]
        if (num / denom) >= pca_energy:
            break

    return k


def pca_svd(x: np.array, pca_energy=0.98, pc=None) -> np.array:
    """PCA using by SVD.
    @:arg x - input matrix of M x N size (M samples with N features per sample).
    @:arg pca_energy - PCA energy, from 0.0 to 1.0.
    @:arg pc- number of principal components, unsigned integer from 1 to N features per sample
    @:return tuple (x_reduced, u_reduced) where
    x_reduced is a reduced matrix of M x K size (K<=N),
    and u_reduced is a matrix of transformation x -> x_reduced."""

    # Mean normalization and feature scaling
    x_std = StandardScaler().fit_transform(x)

    # Calc covariance matrix of M x M size
    cov_mat = np.cov(x_std.T)

    # Calc SVD
    u, s, v = np.linalg.svd(cov_mat)

    # Calc K principal components
    if not pc:
        pc = __calc_pc__(s, pca_energy)

    # Reduce N x M matrix to N x K (K <= M)
    u_reduced = u[:, :pc]
    x_reduced = np.dot(x_std, u_reduced)

    return x_reduced, u_reduced


def pca_eig(x: np.array, pca_energy=0.98, pc=None) -> np.array:
    """PCA using by eigenvector.
    @:arg x - input matrix of M x N size (M samples with N features per sample).
    @:arg pca_energy - PCA energy, from 0.0 to 1.0.
    @:arg pc- number of principal components, unsigned integer from 1 to N features per sample
    @:return tuple (x_reduced, u_reduced) where
    x_reduced is a reduced matrix of M x K size (K<=N),
    and u_reduced is a matrix of transformation x -> x_reduced."""

    # Mean normalization and feature scaling
    x_std = StandardScaler().fit_transform(x)

    # Calc covariance matrix of M x M size
    cov_mat = np.cov(x_std.T)

    # Calc eigenvectors and eigenvalues of covariance matrix
    eig_val_sc, eig_vec_sc = np.linalg.eig(cov_mat)

    # Calc K principal components
    if not pc:
        pc = __calc_pc__(-np.sort(-eig_val_sc), pca_energy)

    # Reduce N x M matrix to N x K (K <= M)
    u_reduced = eig_vec_sc[:, :pc]
    x_reduced = np.dot(x_std, u_reduced)

    return x_reduced, u_reduced
