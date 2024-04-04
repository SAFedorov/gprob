import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError


def cholesky_inv(mat):
    """Inverts the positive-definite symmetric matrix `mat` using Cholesky 
    decomposition. A bit faster than `linalg.inv` and gives a bit smaller error. 
    """
    
    ltr, _ = sp.linalg.cho_factor(mat, check_finite=False, lower=True)
    ltinv = sp.linalg.solve_triangular(ltr, np.eye(ltr.shape[0]), 
                                       check_finite=False, lower=True)
    return ltinv.T @ ltinv


def fisher(cov, dm, dcov):
    """Calculates the Fisher information matrix of an n-dimensional normal
    distribution depending on k parameters.
    
    Args:
        cov: Covariance matrix, (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to the 
            parameters, (k, n, n).

    Returns:
        Fisher information matrix, (k, k).
    """

    # See Eq. 6. in https://arxiv.org/abs/1206.0730v1
    # "Theoretical foundation for CMA-ES from information geometric perspective"

    cov_inv = cholesky_inv(cov)
    prod1 = cov_inv @ dcov

    # Does the same as prod2 = np.einsum('kij, lji -> kl', prod1, prod1), 
    # but faster for large numbers of parameters.
    k, n, _ = dcov.shape
    prod1_flat = np.reshape(prod1, (k, n**2))
    prod1tr_flat = np.reshape(np.transpose(prod1, axes=(0, 2, 1)), (k, n**2))
    prod2 = prod1tr_flat @ prod1_flat.T

    return dm @ cov_inv @ dm.T + 0.5 * prod2


def dlogp(x, m, cov, dm, dcov):
    """Calculates the derivatives of the logarithmic probability density of 
    an n-dimensional normal distribution depending on k parameters with 
    respect to the parameters.

    Args:
        x: Sample value, (n,).
        m: Mean vector, (n,).
        cov: Covariance matrix, (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to the 
            parameters, (k, n, n).

    Returns:
        The gradient vector of the natural logarithm of the probability density 
        at `x` - an array with the shape (k,).
    """

    cov_inv = cholesky_inv(cov)
    dnorm = -0.5 * np.einsum("ij, kij -> k", cov_inv, dcov)
    y = cov_inv @ (x - m)
    
    return 0.5 * y.T @ dcov @ y + dm @ y + dnorm


def d2logp(x, m, cov, dm, dcov, d2m, d2cov):
    """Calculates the second derivatives of the logarithmic probability density 
    of an n-dimensional normal distribution depending on k parameters with 
    respect to the parameters.

    Args:
        x: Sample value, (n,).
        m: Mean vector, (n,).
        cov: Covariance matrix, (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to the 
            parameters, (k, n, n).
        d2m: Second derivatives of the mean vector with respect to the 
            parameters, (k, k, n)
        d2cov: Second derivatives of the covariance matrix with respect to 
            the parameters, (k, k, n, n).
    
    Returns:
        The Hessian of the natural logarithm of the probability density 
        at `x` - an array with the shape (k, k).
    """

    k, n, _ = dcov.shape

    cov_inv = cholesky_inv(cov)
    y = cov_inv @ (x - m)

    term1 = -dm @ cov_inv @ dm.T
    term2 = d2m @ y

    prod1 = cov_inv @ dcov

    # The three lines below are an optimized version of
    # prod2 = prod1 @ (0.5 * np.eye(n) - np.outer(y, (x - m)))
    # term3 = np.einsum('kij, lji -> kl', prod1, prod2)
    prod1_flat = np.reshape(prod1, (k, n**2))
    prod1tr_flat = np.reshape(np.transpose(prod1, axes=(0, 2, 1)), (k, n**2))
    term3 = 0.5 * prod1tr_flat @ prod1_flat.T - ((x-m) @ prod1) @ (prod1 @ y).T

    term4 = 0.5 * np.einsum('klij, ij -> kl', d2cov, np.outer(y, y) - cov_inv)
    term5 = - (dm @ cov_inv) @ (dcov @ y).T

    return term1 + term2 + term3 + term4 + term5 + term5.T


# ---------- functions supporting batching ----------

def logp(x, m, cov):
    """Calculates the logarithmic probability density of an n-dimensional normal
    distribution at the sample value
    
    Args:
        x: The sample(s) at which the likelihood is evaluated. Should be a 
            scalar or an array with the shape (ns,), (n,) or (ns, n), where ns 
            is the number of samples and n is the dimension of the distribution.
        m: The mean vector of the distribution. A scalar or an (n,) array.
        cov: The covariance matrix of the distribution, a scalar or a (n, n) 
            2d array.
        
    Returns:
        The value of logp, or an array of values for each of the input samples.
    """

    x = np.asanyarray(x)
    m = np.asanyarray(m)

    dd = 1 if m.ndim == 0 else len(m)  # distribution dimension

    if dd == 1:
        sd = 1 if x.ndim <= 1 else x.shape[-1]  # sample dimension
    else:
        sd = 1 if x.ndim == 0 else x.shape[-1]
    
    if sd != dd:
        raise ValueError(f"The dimension of the sample vector ({sd}) does not "
                         f"match the dimension of the distribution ({dd}).")
    
    if dd == 1:
        return logp_sc(x, m, cov)
    
    try:
        return logp_cho(x, m, cov)
    except LinAlgError:
        return logp_lstsq(x, m, cov)


def logp_sc(x, mu, sigmasq):
    """logp for scalar inputs."""

    eps = np.finfo(float).eps
    
    if sigmasq < eps:
        # Degenerate case. By our convention, 
        # logp is 1 if x==mu, and -inf otherwise.

        match_idx = np.abs(x - mu) < eps
        llk = np.ones_like(x)
        return np.where(match_idx, llk, float("-inf"))

    return -(x-mu)**2 / (2 * sigmasq) - 0.5 * np.log(2 * np.pi * sigmasq)


def logp_cho(x, m, cov):
    """logp implemented via Cholesky decomposition. Fast for positive-definite 
    covariance matrices, raises LinAlgError for degenerate covariance matrices.
    """

    ltr, _ = sp.linalg.cho_factor(cov, check_finite=False, lower=True)
    z = sp.linalg.solve_triangular(ltr, (x - m).T, 
                                   check_finite=False, lower=True)
    
    rank = cov.shape[0]  # Since the factorization suceeded, the rank is full.
    log_sqrt_det = np.sum(np.log(np.diagonal(ltr)))
    norm = np.log(np.sqrt(2 * np.pi)) * rank + log_sqrt_det

    return -0.5 * np.einsum("i..., i... -> ...", z, z) - norm


def logp_lstsq(x, m, cov):
    """logp implemented via singular value decomposition. Works for arbitrary
    covariance matrices.
    """

    y, _, rank, sv = np.linalg.lstsq(cov, (x - m).T, rcond=None)
    sv = sv[:rank]  # Selects only non-zero singular values.

    norm = np.log(np.sqrt(2 * np.pi)) * rank + np.sum(np.log(sv))
    llk = -0.5 * np.einsum("...i, i... -> ...", x, y) - norm  # log likelihoods

    if rank == cov.shape[0]:
        # The covariance matrix has full rank, all solutions must be good.
        return llk
    
    # Otherwise checks the residual errors.
    delta = (y.T - (x - m)) 
    res = np.einsum("...i, ...i -> ...", delta, delta)
    eps = np.finfo(float).eps * cov.shape[0] 
    valid_idx = np.abs(res) < eps

    return np.where(valid_idx, llk, float("-inf"))