import numpy as np
from numpy.linalg import LinAlgError

from . import emaps
from .emaps import ElementaryMap

from .func import logp


class ConditionError(Exception):
    """Error raised for incompatible conditions."""
    pass


class Normal:
    """Array of normal random variables."""

    __slots__ = ("emap", "b", "size", "shape", "ndim")
    __array_ufunc__ = None

    def __init__(self, emap, b): # TODO: Change to Normal(mu, emap)? -------------------
        if not isinstance(emap, ElementaryMap):
            emap = ElementaryMap(emap)

        if emap.vshape != b.shape:
            raise ValueError(f"The shapes of the map ({emap.vshape}) and "
                             f"the mean ({b.shape}) do not agree.")
        self.emap = emap
        self.b = b

        self.size = b.size
        self.shape = b.shape
        self.ndim = b.ndim

    @property
    def real(self):
        return Normal(self.emap.real, self.b.real)
    
    @property
    def imag(self):
        return Normal(self.emap.imag, self.b.imag)
    
    @property
    def T(self):
        return Normal(self.emap.transpose(), self.b.T)

    @property
    def _a2d(self): return self.emap.a.reshape((self.emap.a.shape[0], -1))
    
    @property
    def _b1d(self): return self.b.reshape((self.b.size,))

    def __repr__(self):
        def rpad(sl):
            max_len = max(len(l) for l in sl)
            return [l.ljust(max_len, " ") for l in sl]
        
        csn = self.__class__.__name__

        meanstr = str(self.mean())
        varstr = str(self.var())

        if "\n" not in meanstr:
            return (f"{csn}(mean={meanstr}, var={varstr})")
        
        meanln = rpad(meanstr.splitlines())
        varln = rpad(varstr.splitlines())

        start = f"{csn}(mean="
        mid = ", var="
        end = ")"
        
        ln = [start + meanln[0] + mid + varln[0] + end]
        ln.extend((" " * len(start)) + ml + (" " * len(mid)) + vl
                  for ml, vl in zip(meanln[1:], varln[1:]))

        return "\n".join(ln)

    def __len__(self):
        return len(self.b)

    def __neg__(self):
        return Normal(-self.emap, -self.b)

    def __add__(self, other):
        if isinstance(other, Normal):
            return Normal(self.emap + other.emap, self.b + other.b)
        
        b = self.b + other
        em = self.emap.broadcast_to(b.shape)
        return Normal(em, b)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Normal):
            return Normal(self.emap + (-other.emap), self.b - other.b)
        
        b = self.b - other
        em = self.emap.broadcast_to(b.shape)
        return Normal(em, b)
    
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, Normal):
            # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
            # for  x = <x> + dx  and  y = <y> + dy.

            b = self.b * other.b
            em = self.emap * other.b + other.emap * self.b
            return Normal(em, b)
        
        return Normal(self.emap * other, self.b * other)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Normal):
            # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
            # for  x = <x> + dx  and  y = <y> + dy.

            b = self.b / other.b
            em = self.emap / other.b + other.emap * ((-self.b) / other.b**2)
            return Normal(em, b)
        
        return Normal(self.emap / other, self.b / other)
    
    def __rtruediv__(self, other):
        # Linearized fraction  x/y = <x>/<y> - dy<x>/<y>^2,
        # for  x = <x>  and  y = <y> + dy.
        # Only need the case when `other` is not a normal variable.
        
        b = other / self.b
        em = self.emap * ((-other) / self.b**2)
        return Normal(em, b)
    
    def __pow__(self, other):
        if isinstance(other, Normal):
            # x^y = <x>^<y> + dx <y> <x>^(<y>-1) + dy ln(<x>) <x>^<y>
            
            b = self.b ** other.b
            em1 = self.emap * (other.b * self.b ** np.where(other.b, other.b-1, 1.))
            em2 = other.emap * (np.log(np.where(self.b, self.b, 1.)) * b)
            return Normal(em1 + em2, b)
        
        other = np.asanyarray(other)  # a number will be subtracted from it
        b = self.b ** other 
        em = self.emap * (self.b ** np.where(other, other-1, 1.))
        return Normal(em, b)

    def __rpow__(self, other):
        # x^y = <x>^<y> + dy ln(<x>) <x>^<y>
        # Only need the case when `other` is not a normal variable.

        b = other ** self.b
        em = self.emap * (np.log(np.where(other, other, 1.)) * b)
        return Normal(em, b)

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
        return Normal(self.emap[key], self.b[key])

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
        emv, emc = emaps.complete((self.emap, cond.emap))

        # The calculation is performed on flattened arrays.
        av = emv.a.reshape((emv.a.shape[0], -1))
        ac = emc.a.reshape((emc.a.shape[0], -1))
        
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
        new_a = np.reshape(new_a, emv.a.shape)
        new_b = np.reshape(new_b, self.b.shape)

        return Normal(ElementaryMap(new_a, emv.elem), new_b)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""

        other = asnormal(other)

        if self.ndim > 1 or other.ndim > 1:
            raise ValueError("& is only applicable to 0- and 1-d arrays.")
        
        b = np.concatenate([self._b1d, other._b1d], axis=0)
        em = emaps.join([self.emap, other.emap])
        return Normal(em, b)  
    
    def __rand__(self, other):
        """Combines two random variables into one vector.""" # TODO: add test cases for this

        other = asnormal(other)  # other is always a constant

        if self.ndim > 1 or other.ndim > 1:
            raise ValueError("& is only applicable to 0- and 1-d arrays.")
        
        b = np.concatenate([other._b1d, self._b1d], axis=0)
        em = emaps.join([other.emap, self.emap])
        return Normal(em, b)

    # ---------- array methods ----------

    def conj(self):
        return Normal(self.emap.conj(), self.b.conj())
    
    def cumsum(self, axis=None, dtype=None):    
        b = self.b.cumsum(axis, dtype=dtype)
        em = self.emap.cumsum(axis, dtype=dtype)
        return Normal(em, b)
    
    def reshape(self, newshape, order="C"):
        b = self.b.reshape(newshape, order=order)
        em = self.emap.reshape(newshape, order=order)
        return Normal(em, b)
    
    def transpose(x, axes=None):
        b = x.b.transpose(axes)
        em = x.emap.transpose(axes)
        return Normal(em, b)

    # ---------- probability-related methods ----------

    def mean(self):
        """Mean"""
        return self.b

    def var(self):
        """Variance"""
        a = self.emap.a

        if np.iscomplexobj(a):
            cvar = np.einsum("i..., i... -> ...", a.conj(), a)
            return np.real(cvar)
        
        return np.einsum("i..., i... -> ...", a, a)
    
    def cov(self):
        """Covariance"""
        a_ = self._a2d

        if np.iscomplexobj(a_):
            return a_.T.conj() @ a_

        return a_.T @ a_
    
    def sample(self, n=None):
        """Samples the random variable `n` times."""
        # n=None returns scalar output

        if n is None:
            nshape = tuple()
        else:
            nshape = (n,)
        
        a = self._a2d
        r = np.random.normal(size=(*nshape, a.shape[0]))
        return (r @ a + self._b1d).reshape((*nshape, *self.shape))
    
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


def join(args): # TODO: remove
    """Combines several random (and possibly deterministic) variables
    into one vector."""

    if isinstance(args, Normal):
        return args

    nvs = [asnormal(v) for v in args]
    em = emaps.join([v.emap for v in nvs])
    b = np.concatenate([v.b.reshape((v.b.size,)) for v in nvs])

    return Normal(em, b)


def asnormal(v):
    if isinstance(v, Normal):
        return v

    # v is a number or array
    b = np.asanyarray(v)
    em = ElementaryMap(np.zeros((0, *b.shape)))
    return Normal(em, b)


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