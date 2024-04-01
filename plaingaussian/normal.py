import numpy as np
from numpy.linalg import LinAlgError

from .elementary import (Elementary, add_maps, complete_maps, 
                         join_maps, u_join_maps)
from .func import logp

# TODO: sampling would be very slow for large-aspect-ratio variables [[a0, a1, a2, a3, .. an]] for n >> 1
# TODO: remove normal-normal arithmetic operations
# TODO: add power operator


class ConditionError(Exception):
    """Error raised for incompatible conditions."""
    pass


class Normal:

    __slots__ = ("a", "b", "iids", "size", "shape", "ndim")
    __array_ufunc__ = None

    def __init__(self, a, b, iids=None):
        if a.shape[1:] != b.shape:
            raise ValueError(f"The shapes of `a` ({a.shape}) and "
                             f"`b` ({b.shape}) do not agree.")

        self.a = a
        self.b = b

        self.size = b.size
        self.shape = b.shape
        self.ndim = b.ndim

        if iids is None:
            iids = Elementary.create(a.shape[0])
        elif len(iids) != a.shape[0]:
            raise ValueError(f"The length of iids ({len(iids)}) does not match "
                             f"the outer dimension of `a` ({a.shape[0]}).")

        self.iids = iids  # Dictionary {id -> index, ...}

    @property
    def _a2d(self): return self.a.reshape((self.a.shape[0], self.b.size))
    
    @property
    def _b1d(self): return self.b.reshape((self.b.size,))

    def __repr__(self):
        if self.size == 0:
            return "" 
        
        if self.size == 1:
            return f"~ normal({self.mean():0.3g}, {self.var():0.3g})"
        
        return f"~ normal(..., shape={self.shape})"
    
    def __len__(self):
        return len(self.b)

    def __neg__(self):
        return Normal(-self.a, -self.b, self.iids)

    def __add__(self, other):
        if isinstance(other, Normal):
            a, iids = add_maps((self.a, self.iids), (other.a, other.iids))
            return Normal(a, self.b + other.b, iids)
        
        b = self.b + other
        a = broadcast_a(self.a, b)
        return Normal(a, b, self.iids)

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        if isinstance(other, Normal):
            a, iids = add_maps((self.a, self.iids), (-other.a, other.iids))
            return Normal(a, self.b - other.b, iids)
        
        b = self.b - other
        a = broadcast_a(self.a, b)
        return Normal(a, b, self.iids)
    
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, Normal):
            # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
            # for  x = <x> + dx  and  y = <y> + dy.

            a, iids = add_maps((self.a * other.b, self.iids),
                               (other.a * self.b, other.iids))
            b = self.b * other.b
            return Normal(a, b, iids)
        
        b = self.b * other 
        a = other * broadcast_a(self.a, b)
        return Normal(a, b, self.iids)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Normal):
            # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
            # for  x = <x> + dx  and  y = <y> + dy.

            a, iids = add_maps((self.a / other.b, self.iids),
                               (other.a * (-self.b) / other.b**2, other.iids))
            b = self.b / other.b
            return Normal(a, b, iids)
        
        b = self.b / other 
        a = broadcast_a(self.a, b) / other
        return Normal(a, b, self.iids)
    
    def __rtruediv__(self, other):
        # Linearized fraction  x/y = <x>/<y> - dy<x>/<y>^2,
        # for  x = <x>  and  y = <y> + dy.
        # Only need co cover the case when `other` is a number or sequence.
        
        b = other / self.b
        a = broadcast_a(self.a, b) * (-other) / self.b**2
        return Normal(a, b, self.iids)
    
    # TODO:
    #def __matmul__(self, other):
    #    pass

    #def __rmatmul__(self, other):
    #    return Normal(other @ self.a, other @ self.b, self.iids)

    def __getitem__(self, key):
        a = self.a[:, key]
        b = self.b[key]
        return Normal(a, b, self.iids)

    def __setitem__(self, key, value):
        if self.iids == value.iids:
            self.a[:, key] = value.a
            self.b[key] = value.b
        else:
            raise ValueError("The iids of the assignment target and the operand"
                             " must be the same to assign at an index.")

    def __or__(self, observations: dict):
        """Conditioning operation.
        
        Args:
            observations: A dictionary of observations in the format 
                {`variable`: `value`, ...}, where `variable`s are normal 
                variables, and `value`s can be constants or normal variables.
        
        Returns:
            Conditional normal variable.

        Raises:
            ConditionError when the observations are incompatible.
        """

        cond = join([k-v for k, v in observations.items()])

        # The calculation is performed on flattened arrays.
        av, ac, union_iids = complete_maps((self._a2d, self.iids),
                                           (cond._a2d, cond.iids))
        
        u, s, vh = np.linalg.svd(ac, compute_uv=True)
        tol = np.finfo(float).eps * np.max(ac.shape)
        snz = s[s > (tol * np.max(s))]  # non-zero singular values
        r = len(snz)  # rank of ac

        u = u[:, :r]
        vh = vh[:r] 

        m = u @ ((vh @ (-cond._b1d)) / snz)
        # lstsq solution of m @ ac = -cond.b

        if r < cond.size:
            nsq = cond._b1d @ cond._b1d
            d = (cond._b1d + m @ ac)  # residual

            if nsq > 0 and (d @ d) > (tol**2 * nsq):
                raise ConditionError("The conditions could not be satisfied. "
                                     f"Got {(d @ d):0.3e} for the residual and "
                                     f"{nsq:0.5e} for |bc|**2.")
                # nsq=0 is always solvable

        proj = (vh.T / snz) @ (u.T @ av) 
        # lstsq solution of ac @ proj = av that projects the column vectors 
        # of av on the subspace spanned by the constraints.

        new_a = av - ac @ proj
        new_b = self._b1d + m @ av

        # Shaping back
        new_a = np.reshape(new_a, (av.shape[0], *self.a.shape[1:]))
        new_b = np.reshape(new_b, self.b.shape)

        return Normal(new_a, new_b, union_iids)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""

        if not isinstance(other, Normal):
            other = asnormal(other)

        if self.b.ndim > 1 or other.b.ndim > 1:
            raise ValueError("& operation is only applicable to 0- and 1-d arrays.")
        
        cat_a, iids = join_maps((self._a2d, self.iids),(other._a2d, other.iids))
        cat_b = np.concatenate([self._b1d, other._b1d], axis=0)

        return Normal(cat_a, cat_b, iids)  
    
    def __rand__(self, other):
        return self & other

    def mean(self):
        """Mean"""
        return self.b

    def var(self):
        """Variance"""
        var = np.einsum("ij, ij -> j", self._a2d, self._a2d)
        return var.reshape(self.shape)
    
    def cov(self):
        """Covariance"""
        a_ = self._a2d
        return a_.T @ a_
    
    def sample(self, n=None):
        """Samples the random variable `n` times."""
        # n=None returns scalar output

        if n is None:
            nshape = tuple()
        else:
            nshape = (n,)
        
        r = np.random.normal(size=(*nshape, len(self.iids)))
        return (r @ self._a2d + self._b1d).reshape((*nshape, *self.shape))  
    
    def logp(self, x):
        """Log likelihood of a sample.
    
        Args:
            x: Sample value or a sequence of sample values.

        Returns:
            Natural logarithm of the probability density at the sample value - 
            a single number for single sample inputs, and an array for sequence 
            inputs.
        """
        return logp(x, self._b1d, self.cov())


def join(args):
    """Combines several of random (and possibly deterministic) variables
    into one vector."""

    if isinstance(args, Normal):
        return args

    nvs = [asnormal(v) for v in args]
    ops = [(v._a2d, v.iids) for v in nvs]
    a, iids = u_join_maps(ops)
    b = np.concatenate([np.array(v.b, ndmin=1) for v in nvs])

    return Normal(a, b, iids)


def asnormal(v):
    if isinstance(v, Normal):
        return v

    # v is a number or sequence of numbers
    return Normal(a=np.array([]), b=np.array(v), iids={})


def normal(mu=0, sigmasq=1, size=1):
    """Creates a new normal random variable.
    
    Args:
        mu: scalar or vector mean value
        sigmasq: scalar variance or covariance matrix

    Returns:
        Normal random variable, scalar or vector.
    """

    sigmasq = np.array(sigmasq)
    mu = np.array(mu)

    if sigmasq.size == 1:

        sigmasq = sigmasq.reshape((1,))
        mu = mu.reshape(tuple())  # makes scalar
        
        if size == 1:
            # Single scalar variable
            if sigmasq < 0:
                raise ValueError("Negative scalar sigmasq")
            
            return Normal(np.sqrt(sigmasq), mu)
        
        # Vector of independent identically-distributed variables.
        return Normal(np.sqrt(sigmasq) * np.eye(size, size), mu * np.ones(size))

    mu = np.broadcast_to(mu, (sigmasq.shape[0],))
        
    try:
        atr = np.linalg.cholesky(sigmasq)
        return Normal(atr.T, mu)
    except LinAlgError:
        # The covariance matrix is not strictly positive-definite.
        pass

    # To handle the positive-semidefinite case, do the orthogonal decomposition. 
    eigvals, eigvects = np.linalg.eigh(sigmasq)  # sigmasq = V D V.T

    if (eigvals < 0).any():
        raise ValueError("Negative eigenvalue in sigmasq matrix.")
    
    atr = eigvects @ np.diag(np.sqrt(eigvals))

    return Normal(atr.T, mu)


def broadcast_a(a, b):
    if a.ndim != b.ndim + 1:
        a = np.broadcast_to(a, (a.shape[0], *b.shape))
    return a