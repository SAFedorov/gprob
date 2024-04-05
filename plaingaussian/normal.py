import numpy as np
from numpy.linalg import LinAlgError

from .elementary import (Elementary, add_maps, complete_maps, 
                         join_maps, u_join_maps)
from .func import logp


class ConditionError(Exception):
    """Error raised for incompatible conditions."""
    pass


class Normal:
    """Array of normally-distributed random variables, represented as
    
    x[...] = b[...] + sum_k a[i...] xi[i],
    
    where and `xi`s are elementary Gaussian variables that are independent and 
    identically-distributed, `xi`[i] ~ N(0, 1) for all i, and ... is a 
    multi-dimensional index.
    """

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

        self.iids = iids  # Dictionary of elementary variables {id -> k, ...}

    @property
    def _a2d(self): return self.a.reshape((self.a.shape[0], self.b.size))
    
    @property
    def _b1d(self): return self.b.reshape((self.b.size,))

    def __repr__(self):
        csn = self.__class__.__name__

        meanstr = str(self.mean())
        varstr = str(self.var())

        if "\n" not in meanstr:
            return (f"{csn}(mean={meanstr}, var={varstr})")
        
        return (f"{csn}(mean=\n{meanstr},\nvar=\n{varstr})")

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

            b = self.b * other.b
            a, iids = add_maps((unsqueeze_a(self.a, b) * other.b, self.iids),
                               (unsqueeze_a(other.a, b) * self.b, other.iids))
            return Normal(a, b, iids)
        
        b = self.b * other 
        a = other * unsqueeze_a(self.a, b)
        return Normal(a, b, self.iids)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Normal):
            # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
            # for  x = <x> + dx  and  y = <y> + dy.

            b = self.b / other.b
            a1 = unsqueeze_a(self.a, b) / other.b
            a2 = unsqueeze_a(other.a, b) * (-self.b) / other.b**2
            a, iids = add_maps((a1, self.iids), (a2, other.iids))
            return Normal(a, b, iids)
        
        b = self.b / other 
        a = unsqueeze_a(self.a, b) / other
        return Normal(a, b, self.iids)
    
    def __rtruediv__(self, other):
        # Linearized fraction  x/y = <x>/<y> - dy<x>/<y>^2,
        # for  x = <x>  and  y = <y> + dy.
        # Only need the case when `other` is not a normal variable.
        
        b = other / self.b
        a = unsqueeze_a(self.a, b) * (-other) / self.b**2
        return Normal(a, b, self.iids)
    
    def __pow__(self, other):
        if isinstance(other, Normal):
            # x^y = <x>^<y> + dx <y> <x>^(<y>-1) + dy ln(<x>) <x>^<y>
            
            b = self.b ** other.b
            a1 = unsqueeze_a(self.a, b) * (other.b * self.b ** np.where(other.b, other.b-1, 1.))
            a2 = unsqueeze_a(other.a, b) * (np.log(np.where(self.b, self.b, 1.)) * b)
            a, iids = add_maps((a1, self.iids), (a2, other.iids))
            return Normal(a, b, iids)
        
        other = np.asanyarray(other)  # because a number will be subtracted from it
        b = self.b ** other 
        gb = other * self.b ** np.where(other, other-1, 1.)
        a = gb * unsqueeze_a(self.a, b)
        return Normal(a, b, self.iids)

    def __rpow__(self, other):
        # x^y = <x>^<y> + dy ln(<x>) <x>^<y>
        # Only need the case when `other` is not a normal variable.

        b = other ** self.b
        a = unsqueeze_a(self.a, b) * (np.log(np.where(other, other, 1.)) * b)
        return Normal(a, b, self.iids)

    def __matmul__(self, other):  # TODO: add tests for this operation
        if isinstance(other, Normal):
            raise NotImplementedError
        
        b = self.b @ other
        a = unsqueeze_a(self.a, b) @ other
        return Normal(a, b, self.iids)

    def __rmatmul__(self, other):
        if isinstance(other, Normal):
            raise NotImplementedError

        b = other @ self.b
        if self.ndim == 1:
            a = (other @ self.a.T).T
        else:
            a = other @ unsqueeze_a(self.a, b)

        return Normal(a, b, self.iids)

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

        if self.ndim > 1 or other.ndim > 1:
            raise ValueError("& is only applicable to 0- and 1-d arrays.")
        
        a, iids = join_maps((self._a2d, self.iids), (other._a2d, other.iids))
        b = np.concatenate([self._b1d, other._b1d], axis=0)
        return Normal(a, b, iids)  
    
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
        
        x = np.asanyarray(x)
        if self.ndim > 1:
            if x.ndim == self.ndim:
                x = x.reshape((x.size,))
            else:
                x = x.reshape((x.shape[0], -1)) 
        return logp(x, self._b1d, self.cov())


def join(args):
    """Combines several random (and possibly deterministic) variables
    into one vector."""

    if isinstance(args, Normal):
        return args

    nvs = [asnormal(v) for v in args]
    ops = [(v._a2d, v.iids) for v in nvs]
    a, iids = u_join_maps(ops)
    b = np.concatenate([v.b.reshape((v.b.size,)) for v in nvs])

    return Normal(a, b, iids)


def asnormal(v):
    if isinstance(v, Normal):
        return v

    # v is a number or sequence of numbers
    return Normal(a=np.array([]), b=np.asanyarray(v), iids={})


def normal(mu=0., sigmasq=1., size=None):
    """Creates a new normal random variable.
    
    Args:
        mu: Mean value, a scalar or an array.
        sigmasq: Scalar variance or matrix covariance.
        size: Optional integer or tuple specifying the size and shape of 
            the variable. Only has an effect with scalar mean and variance. 

    Returns:
        Normal random variable.
    """

    sigmasq = np.asanyarray(sigmasq)
    mu = np.asanyarray(mu)

    if sigmasq.ndim == 0:
        if sigmasq < 0:
            raise ValueError("Negative value for the variance.")
        
        sigma = np.sqrt(sigmasq)
    
        if mu.ndim == 0:
            if not size:
                return Normal(sigma[None], mu)  # expanding sigma to 1d
            elif isinstance(size, int):
                a = sigma * np.eye(size, size)
                return Normal(a, np.broadcast_to(mu, (size,)))
            else:
                b = np.broadcast_to(mu, size)
                a = sigma * np.eye(b.size, b.size).reshape((b.size, *b.shape))
                return Normal(a, b)
        else:
            a = sigma * np.eye(mu.size, mu.size).reshape(mu.size, *mu.shape)
            return Normal(a, mu)
        
    if sigmasq.ndim != 2:
        raise ValueError("Only 0 and 2 dimensional covariance matrices are presently supported.")

    mu = np.broadcast_to(mu, (sigmasq.shape[0],))
        
    try:
        atr = np.linalg.cholesky(sigmasq)
        return Normal(atr.T, mu)
    except LinAlgError:
        # The covariance matrix is not strictly positive-definite.
        pass

    # Handles the positive-semidefinite case using orthogonal decomposition. 
    eigvals, eigvects = np.linalg.eigh(sigmasq)  # sigmasq = V D V.T

    if (eigvals < 0).any():
        raise ValueError("Negative eigenvalue(s) in the covariance matrix.")
    
    atr = eigvects @ np.diag(np.sqrt(eigvals))

    return Normal(atr.T, mu)


def unsqueeze_a(a, b):
    dn = (b.ndim + 1) - a.ndim
    if dn != 0:
        sh = list(a.shape)
        sh[1:1] = (1,) * dn
        a = a.reshape(sh)
    return a


def broadcast_a(a, b):
    dn = (b.ndim + 1) - a.ndim
    if dn != 0:
        sh = list(a.shape)
        sh[1:1] = (1,) * dn
        a = np.broadcast_to(a.reshape(sh), (a.shape[0], *b.shape))
    return a