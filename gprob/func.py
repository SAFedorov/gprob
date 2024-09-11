import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError


def cholesky_inv(mat):
    """Inverts the positive-definite symmetric matrix ``mat`` using Cholesky 
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
        at ``x`` - an array with the shape (k,).
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
        at ``x`` - an array with the shape (k, k).
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


def logp(x, m, cov):
    """Calculates the logarithmic probability density of an n-dimensional normal
    distribution at the sample value
    
    Args:
        x: The sample(s) at which the likelihood is evaluated. Should be a 
            scalar or an array with the shape (ns,), (n,) or (ns, n), where ns 
            is the number of samples and n is the dimension of the distribution.
        m: The mean of the distribution. A scalar or a (n,) - shaped array.
        cov: The covariance of the distribution, a scalar or a (n, n) -
            shaped 2d array.
        
    Returns:
        The value of logp, or an array of values for each of the input samples.
    """

    x = np.asanyarray(x)
    m = np.asanyarray(m)
    
    # Determines the sample size.
    if x.ndim == m.ndim:
        ssz = x.size
    elif x.ndim == m.ndim + 1:
        ssz = 1 if x.ndim == 1 else x.shape[-1]
    else:
        raise ValueError(f"The sample array must have {m.ndim} or {m.ndim+1} "
                         f"dimensions to be compatible with the {m.ndim}-"
                         f"dimensional mean, while now it has {x.ndim} "
                         "dimensions.")
    
    if ssz != m.size:
        raise ValueError(f"The size of the sample vector ({ssz}) does not "
                         f"match the size of the distribution ({m.size}).")
    
    if m.size == 1:
        x = x.reshape(x.shape[:x.ndim - m.ndim])
        m = m.item()
        cov = cov.item()
        return logp_sc(x, m, cov)
    
    try:
        return logp_cho(x, m, cov)
    except LinAlgError:
        return logp_lstsq(x, m, cov)


def logp_sc(x, mu, sigmasq):
    """logp for scalar inputs."""
    
    if sigmasq == 0:
        # Degenerate case. By our convention, 
        # logp is 1 if x==mu, and -inf otherwise.

        match_idx = np.equal(x, mu)
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
    norm = 0.5 * np.log(2 * np.pi) * rank + log_sqrt_det

    return -0.5 * np.einsum("i..., i... -> ...", z, z) - norm


def logp_lstsq(x, m, cov):
    """logp implemented via singular value decomposition. Works for arbitrary
    covariance matrices.
    """

    dx = (x - m)
    y, _, rank, sv = np.linalg.lstsq(cov, dx.T, rcond=None)
    sv = sv[:rank]  # Selects only non-zero singular values.

    norm = 0.5 * np.log(2 * np.pi) * rank + 0.5 * np.sum(np.log(sv))
    llk = -0.5 * np.einsum("...i, i... -> ...", dx, y) - norm  # log likelihoods

    if rank == cov.shape[0]:
        # The covariance matrix has full rank, all solutions must be good.
        return llk
    
    # Otherwise checks the residual errors.
    delta = ((cov @ y).T - dx) 
    res = np.einsum("...i, ...i -> ...", delta, delta)
    eps = np.finfo(y.dtype).eps * cov.shape[0] * (np.max(sv)**2) 
    valid_idx = np.abs(res) < eps

    return np.where(valid_idx, llk, float("-inf"))


class ConditionError(LinAlgError):
    """Error raised in conditioning."""
    pass


def condition(m, a, mc, ac, mask=None):
    """Conditions the vandom variable ``v`` on another random variable, ``c``, 
    being equal to zero. Both variables are defined as Gaussian maps,

    v = m + e @ a,
    c = mc + e @ ac,
    
    where ``e`` are vectors of independent identically-distributed 
    normal random variables with zero mean and unit variance, ``m`` and ``mc`` 
    are the mean vectors and `a` and `ac` are the map matrices.  
    
    Args:
        m: The mean vector of the variable to be conditioned, an (n,) array.
        a: The map matrix of the variable to be conditioned, a (ne, n) 2d array.
        mc: The mean vector of the variable conditioned on, an (nc,) array.
        ac: The map matrix of the variable conditioned on, a (ne, nc) 2d array.
        mask (bool array, optional): A 2d mask with the shape (nc, n), where
            mask[i, j] == False means than the i-th condition does not affect 
            the j-th variable.
        
    Returns:
        Tuple (conditional mean, conditional map matrix)
    """

    try:
        cond_m, cond_a = condition_qr(m, a, mc, ac, mask)
    except LinAlgError:
        if mask is not None:
            raise ConditionError("Masks can only be used with non-degenerate "
                                 "conditions.")
        
        cond_m, cond_a = condition_svd(m, a, mc, ac)
    
    return cond_m, cond_a


def condition_qr(m, a, mc, ac, mask=None):
    """Conditioning using QR or RQ decomposition."""

    def ql(a):
        r, q = sp.linalg.rq(a.T, mode="economic", check_finite=False)
        return q.T, r.T

    def qu(a):
        q, r = sp.linalg.qr(a, mode="economic", check_finite=False)
        return q, r

    if ac.shape[0] < ac.shape[1]:
        raise ConditionError("Conditioning via QR decomposition does not work "
                             "with degenerate constraints. Use SVD instead.")

    qtri = qu
    if mask is not None and not mask[0, -1]:
        # For lower-triangular masks need to use ql decomposition.
        qtri = ql

    q, tri = qtri(ac)

    # Checks if there are zero diagonal elements in the triangular matrix, 
    # which would mean that some colums of `ac` are linearly dependent.
    diatri = np.abs(np.diag(tri))
    tol = np.finfo(tri.dtype).eps
    if (diatri < (tol * np.max(diatri))).any():
        raise ConditionError("Conditioning via QR decomposition does not work "
                             "with degenerate constraints. Use SVD instead.")

    es = sp.linalg.solve_triangular(tri.T, -mc, lower=(qtri is qu),
                                    check_finite=False)

    aproj = (q.T @ a)
    # The projection of the column vectors of `a` on the subspace spanned by 
    # the column vectors of the constraint matrix `ac`.

    if mask is not None:
        aproj = np.where(mask, aproj, 0)

    cond_a = a - q @ aproj
    cond_m = m + es @ aproj

    return cond_m, cond_a


def condition_svd(m, a, mc, ac):
    u, s, vh = np.linalg.svd(ac, compute_uv=True)

    tol = np.finfo(u.dtype).eps * np.max(ac.shape)
    snz = s[s > (tol * np.max(s))]  # non-zero singular values
    r = len(snz)  # rank of ac

    u = u[:, :r]
    vh = vh[:r] 

    el = u @ ((vh @ (-mc)) / snz)
    # lstsq solution of el @ ac = -mc

    if r < mc.size:
        nsq = mc @ mc
        d = (mc + el @ ac)  # residual

        if nsq > 0 and (d @ d) > (tol**2 * nsq):
            raise ConditionError("The conditions could not be satisfied. "
                                    f"Got {(d @ d):0.3e} for the residual and "
                                    f"{nsq:0.5e} for |mu_c|**2.")
            # nsq=0 is always solvable

    cond_a = a - u @ (u.T @ a)
    cond_m = m + el @ a

    return cond_m, cond_a