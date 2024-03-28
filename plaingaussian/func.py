import numpy as np
import scipy as sp
from scipy.linalg import lapack


# Note: the choice of the parameter dimension order is made to fascilitate 
# broadcasting in matrix multiplication.

def cholesky_inv(mat):
    """Inverts the positive-definite symmetric matrix `mat` using Cholesky 
    decomposition. Marginally faster than `linalg.inv`. """

    # TODO: there are scipy distributions where it actually works slower than np.linalg.inv
    # TODO: maybe use higher-level methods from scipy for the decomposition, like chol_factor, it may become faster
    # TODO: check if the upper triangular may contain random data
    
    lt, status = lapack.dpotrf(mat, lower=True)  #TODO: use scipy.linalg.cho_factor instead

    if status != 0:
        raise RuntimeError("Cholesky decomposition failed.")  # TODO: expand the error
    
    ltinv = sp.linalg.solve_triangular(lt, np.eye(lt.shape[0]), 
                                       check_finite=False, lower=True)
    
    return ltinv.T @ ltinv


def fisher(cov, dm, dcov):
    """Calculates the Fisher information matrix of an `n`-dimensional normal
    distribution depending on `k` parameters.
    
    Args:
        cov: Covariance matrix (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to 
            the parameters, (k, n, n).

    Returns:
        Fisher information matrix, (k, k).
    """

    # See Eq. 6. in https://arxiv.org/abs/1206.0730v1
    # "Theoretical foundation for CMA-ES from information geometric perspective"

    cov_inv = cholesky_inv(cov)
    prod1 = cov_inv @ dcov

    # Does the same as prod2 = np.einsum('kij, lji -> kl', prod1, prod1), 
    # in a way that is faster for large numbers of parameters.
    k, n, _ = dcov.shape
    prod1_flat = np.reshape(prod1, (k, n**2))
    prod1tr_flat = np.reshape(np.transpose(prod1, axes=(0, 2, 1)), (k, n**2))
    prod2 = prod1tr_flat @ prod1_flat.T

    return dm @ cov_inv @ dm.T + 0.5 * prod2


def logp(x, m, cov):

    # TODO: ?

    lt, status = lapack.dpotrf(cov, lower=True)

    if status != 0:
        raise RuntimeError("Cholesky decomposition failed.")

    z = sp.linalg.solve_triangular(lt, x - m, check_finite=False, lower=True)
    rank = cov.shape[0]  # If the solution suceeded, the rank is full.
    log_sqrt_det = np.sum(np.log(np.diagonal(lt)))

    norm = np.log(np.sqrt(2 * np.pi)) * rank + log_sqrt_det
    return - z @ z / 2 - norm


def dlogp(x, m, cov, dm, dcov):
    """Calculates the derivatives of the logarithmic probability density with 
    respect to the parameters.

    Args:
        x: Sample value.
        m: Mean vector (n).
        cov: Covariance matrix (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to 
            the parameters, (k, n, n).

    Returns:
        Gradient vector of the natural logarithm of the probability at x, (k).
    """

    cov_inv = cholesky_inv(cov)
    dnorm = -0.5 * np.einsum("ij, kji -> k", cov_inv, dcov)
    y = cov_inv @ (x - m)
    
    return 0.5 * y.T @ dcov @ y + dm @ y + dnorm


def d2logp(x, m, cov, dm, dcov, d2m, d2cov):

    k, n, _ = dcov.shape

    cov_inv = cholesky_inv(cov)
    y = cov_inv @ (x - m)

    term1 = -dm @ cov_inv @ dm.T
    term2 = d2m @ y

    prod1 = cov_inv @ dcov

    # The three lines below do the same as
    # prod2 = prod1 @ (0.5 * np.eye(n) - np.outer(y, (x - m)))
    # term3 = np.einsum('kij, lji -> kl', prod1, prod2) 
    # in a way that is faster for large numbers of parameters.
    prod1_flat = np.reshape(prod1, (k, n**2))
    prod1tr_flat = np.reshape(np.transpose(prod1, axes=(0, 2, 1)), (k, n**2))
    term3 = 0.5 * prod1tr_flat @ prod1_flat.T - ((x-m) @ prod1) @ (prod1 @ y).T

    term4 = 0.5 * np.einsum('klij, ji -> kl', d2cov, np.outer(y, y) - cov_inv)
    term5 = - (dm @ cov_inv) @ (dcov @ y).T

    return term1 + term2 + term3 + term4 + term5 + term5.T