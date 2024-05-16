from builtins import sum as sum_

import numpy as np
from numpy.linalg import LinAlgError

from . import emaps
from .emaps import ElementaryMap

from .func import logp, condition


class Normal:
    """Array of normal random variables."""

    __slots__ = ("emap", "b", "size", "shape", "ndim")
    __array_ufunc__ = None

    def __init__(self, emap, b):
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
    def a(self):
        return self.emap.a
    
    @property
    def iscomplex(self):
        return (np.iscomplexobj(self.emap.a) or np.iscomplexobj(self.b))

    def __repr__(self):
        csn = self.__class__.__name__

        if self.ndim == 0:
            meanstr = f"{self.mean():.8g}"
            varstr = f"{self.var():.8g}"
            # To make the displays of scalars consistent with the display 
            # of array elements.
        else:
            meanstr = str(self.mean())
            varstr = str(self.var())

        if "\n" not in meanstr and "\n" not in varstr:
            return (f"{csn}(mean={meanstr}, var={varstr})")

        meanln = meanstr.splitlines()
        h = "  mean="
        meanln_ = [h + meanln[0]]
        meanln_.extend(" " * len(h) + ln for ln in meanln[1:])

        varln = varstr.splitlines()
        h = "   var="
        varln_ = [h + varln[0]]
        varln_.extend(" " * len(h) + ln for ln in varln[1:])

        return "\n".join([f"{csn}(", *meanln_, *varln_, ")"])

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("Scalar arrays have no lengh.")
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
        
        other = np.asanyarray(other)
        b = self.b ** other
        em = self.emap * (other * (self.b ** np.where(other, other-1, 1.)))
        return Normal(em, b)

    def __rpow__(self, other):
        # x^y = <x>^<y> + dy ln(<x>) <x>^<y>
        # Only need the case when `other` is not a normal variable.

        b = other ** self.b
        em = self.emap * (np.log(np.where(other, other, 1.)) * b)
        return Normal(em, b)

    def __matmul__(self, other):
        if isinstance(other, Normal):
            return self @ other.b + self.b @ other
        return Normal(self.emap @ other, self.b @ other)

    def __rmatmul__(self, other):
        return Normal(other @ self.emap, other @ self.b)

    def __getitem__(self, key):
        return Normal(self.emap[key], self.b[key])
    
    def __setitem__(self, key, value):
        value = asnormal(value)

        if not self.b.flags.writeable:
            self.b = self.b.copy()

        self.b[key] = value.b
        self.emap[key] = value.emap

    def __or__(self, observations: dict):
        """Conditioning operation."""
        return self.condition(observations)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""
        return hstack([self, other])  
    
    def __rand__(self, other):
        """Combines two random variables into one vector."""
        return hstack([other, self])

    # ---------- array methods ----------

    def conjugate(self):
        return Normal(self.emap.conj(), self.b.conj())
    
    def conj(self):
        return self.conjugate()
    
    def cumsum(self, axis=None):    
        b = self.b.cumsum(axis)
        em = self.emap.cumsum(axis)
        return Normal(em, b)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        b = self.b.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        em = self.emap.diagonal(offset=offset, vaxis1=axis1, vaxis2=axis2)
        return Normal(em, b)

    def flatten(self, order="C"):
        b = self.b.flatten(order=order)
        em = self.emap.flatten(order=order)
        return Normal(em, b)
    
    def moveaxis(self, source, destination):
        b = np.moveaxis(self.b, source, destination)
        em = self.emap.moveaxis(source, destination)
        return Normal(em, b)
    
    def ravel(self, order="C"):
        b = self.b.ravel(order=order)
        em = self.emap.ravel(order=order)
        return Normal(em, b)
    
    def reshape(self, newshape, order="C"):
        b = self.b.reshape(newshape, order=order)
        em = self.emap.reshape(newshape, order=order)
        return Normal(em, b)
    
    def sum(self, axis=None, keepdims=False):
        # "where" is absent because its broadcasting is not implemented.
        # "initial" is also not implemented.
        b = self.b.sum(axis, keepdims=keepdims)
        em = self.emap.sum(axis, keepdims=keepdims)
        return Normal(em, b)

    def transpose(self, axes=None):
        b = self.b.transpose(axes)
        em = self.emap.transpose(axes)
        return Normal(em, b)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        b = self.b.trace(offset=offset, axis1=axis1, axis2=axis2)
        em = self.emap.trace(offset, axis1, axis2)
        return Normal(em, b)

    # ---------- probability-related methods ----------

    def condition(self, observations, mask=None):
        """Conditioning operation.
        
        Args:
            observations (Normal or dict):
                A single random normal variable or a dictionary of observations
                of the format 
                {`variable`: `value`, ...}, where `variable`s are normal 
                variables, and `value`s can be numerical constants or 
                random variables. A single normal `variable` is equavalent to 
                {`variable`: `0`}.
            mask (optional): 
                A 2d bool array, in which `mask[i, j] == True` means that 
                the `i`-th condition applies to the `j`-th variable, and
                `False` that it does not.
                In the case when the variables have more than one dimension, 
                the 0th axis of `mask` spans over the 0th axis of each of the 
                conditions, and the 1st axis of `mask` spans over the 0th axis
                of the conditioned variable.
                The mask needs to be generalized upper- or lower- triangular, 
                meaning that there needs to be a set of indices `i0[j]` such 
                that either `mask[i, j] == True` for all `i > i0[j]` and 
                `mask[i, j] == False` for `i < i0[j]`, or `mask[i, j] == True` 
                for all `i < i0[j]` and `mask[i, j] == False` for `i > i0[j]`.
        
        Returns:
            Conditional normal variable.

        Raises:
            ConditionError if the observations are mutually incompatible,
            or if a mask is given with degenerate observations.
        """
        if isinstance(observations, dict):
            obs = [asnormal(k-v) for k, v in observations.items()]
        else:
            obs = [asnormal(observations)]

        if mask is None:
            cond = concatenate([c.ravel() for c in obs])
        else:
            # Concatenates preserving the element order along the 0th axis.

            if self.ndim < 1:
                raise ValueError("The variable must have at least one "
                                 "dimension to be conditioned with a mask.")

            if any(c.ndim < 1 for c in obs):
                raise ValueError("All conditions must have at least one "
                                 "dimension to be compatible with masking.")
            
            obs = [c.reshape((c.shape[0], -1)) for c in obs]
            cond = concatenate(obs, axis=1).ravel()

            k = sum_(c.shape[1] for c in obs)
            l = 1 if self.ndim == 1 else np.prod(self.shape[1:])
            ms = mask.shape

            mask = np.broadcast_to(mask[:, None, :, None], (ms[0], k, ms[1], l))
            mask = mask.reshape((ms[0] * k, ms[1] * l))

        emv, emc = emaps.complete((self.emap, cond.emap))

        a = emv.a2d
        m = self.b.ravel()
        if self.iscomplex:
            # Doubles the dimension preserving the triangular structure.
            a = np.stack([a.real, a.imag], axis=-1).reshape(a.shape[0], -1)
            m = np.stack([m.real, m.imag], axis=-1).ravel()

            if mask is not None:
                mask = np.stack([mask, mask], axis=-1)
                mask = mask.reshape(mask.shape[0], -1)

        ac = emc.a
        mc = cond.b
        if cond.iscomplex:
            # Doubles the dimension preserving the triangular structure.
            ac = np.stack([ac.real, ac.imag], axis=-1).reshape(ac.shape[0], -1)
            mc = np.stack([mc.real, mc.imag], axis=-1).ravel()

            if mask is not None:
                mask = np.stack([mask, mask], axis=-2)
                mask = mask.reshape(-1, mask.shape[-1])

        new_b, new_a = condition(m, a, mc, ac, mask)

        if self.iscomplex:
            # Converting back to complex.
            new_a = new_a[:, 0::2] + 1j * new_a[:, 1::2]
            new_b = new_b[0::2] + 1j * new_b[1::2]

        # Shaping back.
        new_a = np.reshape(new_a, emv.a.shape)
        new_b = np.reshape(new_b, self.b.shape)

        return Normal(ElementaryMap(new_a, emv.elem), new_b)

    def mean(self):
        """Mean"""
        return self.b

    def variance(self):
        """Variance, `<(x-<x>)(x-<x>)^*>` where `*` is complex conjugation."""
        # Note: has an alias "var"
        
        a = self.emap.a        
        return np.real(np.einsum("i..., i... -> ...", a, a.conj()))
    
    def var(self):
        """Variance, `<(x-<x>)(x-<x>)^*>` where `*` is complex conjugation."""
        # Note: an alias for "variance"
        return self.variance()

    def covariance(self):
        """Covariance. 
        
        For a vector variable `x` returns the matrix `C = <(x-<x>)(x-<x>)^H>`, 
        where `H` is conjugate transpose.

        For a general array `x` returns the array `C` with twice the number of 
        dimensions of `x` and the components
        `C[ijk... lmn...] = <(x[ijk..] - <x>) (x[lmn..] - <x>)*>`, 
        where the indices `ijk...` and `lmn...` run over the components of `x`,
        and `*` is complex conjugation.
        """
        # Note: has an alias "cov"

        a = self.emap.a2d
        cov2d = a.T @ a.conj()
        return cov2d.reshape(self.shape * 2)
    
    def cov(self):
        """Covariance. 
        
        For a vector variable `x` returns the matrix `C = <(x-<x>)(x-<x>)^H>`, 
        where `H` is conjugate transpose.

        For a general array `x` returns the array `C` with twice the number of 
        dimensions of `x` and the components
        `C[ijk... lmn...] = <(x[ijk..] - <x>) (x[lmn..] - <x>)*>`, 
        where the indices `ijk...` and `lmn...` run over the components of `x`,
        and `*` is complex conjugation.
        """
        # Note: an alias for "covariance"
        return self.covariance()
    
    def sample(self, n=None):
        """Samples the random variable `n` times."""
        # n=None returns scalar output

        if n is None:
            nshape = tuple()
        else:
            nshape = (n,)
        
        a = self.emap.a2d
        r = np.random.normal(size=(*nshape, a.shape[0]))
        return (r @ a + self.b.ravel()).reshape((*nshape, *self.shape))
    
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
            # Flattens the sample values.

            if x.ndim == self.ndim:
                x = x.reshape((x.size,))
            else:
                x = x.reshape((x.shape[0], -1))

        m = self.b.ravel()
        a = self.emap.a2d
        if self.iscomplex:
            # Converts to real by doubling the space size.
            x = np.hstack([x.real, x.imag])
            m = np.hstack([m.real, m.imag])
            a = np.hstack([a.real, a.imag])
        
        cov = a.T @ a 
        return logp(x, m, cov)


def asnormal(v):
    if isinstance(v, Normal):
        return v

    # v is a number or array
    b = np.asanyarray(v)
    em = ElementaryMap(np.zeros((0, *b.shape), dtype=b.dtype))
    return Normal(em, b)


def normal(mu=0., sigmasq=1., size=None):
    """Creates a new normal random variable.
    
    Args:
        mu: Mean value, a scalar or an array.
        sigmasq: Scalar variance or matrix covariance.
        size: Optional integer or sequence of integers specifying the shape of 
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
                b = np.broadcast_to(mu, (size,))
                a = sigma * np.eye(size, size, dtype=sigma.dtype)
                return Normal(a, b)
            else:
                b = np.broadcast_to(mu, size)
                a = sigma * np.eye(b.size, b.size, dtype=sigma.dtype)
                a = a.reshape((b.size, *b.shape))
                return Normal(a, b)
        else:
            a = sigma * np.eye(mu.size, mu.size, dtype=sigma.dtype)
            a = a.reshape((mu.size, *mu.shape))
            return Normal(a, mu)
        
    if sigmasq.ndim % 2 != 0:
        raise ValueError("The number of the dimensions of the covariance "
                         f"matrix must be even, while it is {sigmasq.ndim}.")
    
    vnd = sigmasq.ndim // 2
    if sigmasq.shape[:vnd] != sigmasq.shape[vnd:]:
        raise ValueError("The first and the second halves of the covaraince "
                         "matrix shape must be identical, while they are "
                         f"{sigmasq.shape[:vnd]} and {sigmasq.shape[vnd:]}.")
    
    vshape = sigmasq.shape[:vnd]
    mu = np.broadcast_to(mu, vshape)
    sigmasq2d = sigmasq.reshape((mu.size, mu.size))
        
    try:
        a2dtr = _safer_cholesky(sigmasq2d)
        a = np.reshape(a2dtr.T, (mu.size,) + vshape)
        return Normal(a, mu)
    except LinAlgError:
        # The covariance matrix is not strictly positive-definite.
        pass

    # Handles the positive-semidefinite case using unitary decomposition. 
    eigvals, eigvects = np.linalg.eigh(sigmasq2d)  # sigmasq = V D V.H

    atol = len(eigvals) * np.max(np.abs(eigvals)) * np.finfo(eigvals.dtype).eps
    if (eigvals < -atol).any():
        raise ValueError("Not all eigenvalues of the covariance matrix are "
                         f"non-negative: {eigvals}.")
    
    eigvals[eigvals < 0] = 0.
    a2dtr = eigvects * np.sqrt(eigvals)
    a = np.reshape(a2dtr.T, (mu.size,) + vshape)

    return Normal(a, mu)


def _safer_cholesky(x):
    ltri = np.linalg.cholesky(x)

    d = np.diagonal(ltri)
    atol = 100 * len(d) * np.finfo(d.dtype).eps * np.max(d**2)
    if (d**2 < atol).any():
        raise LinAlgError("The input matrix seems to be degenerate.")
    
    return ltri


def covariance(*args):
    """Covariance. For a single variable `x` the same as `x.covariance()`.
    
    For two variables `x` and `y` it is defined in an element-wise manner as 
    `<(x-<x>) (y-<y>)^*>`, where `*` is complex conjugation.

    For two arrays `x` and `y` the function returns the array `C` with 
    the number of dimensions equal to the sum of the dimensions of `x` and `y` 
    and the components
    `C[ijk... lmn...] = <(x[ijk..] - <x>) (y[lmn..] - <y>)*>`, 
    where the indices `ijk...` and `lmn...` run over the components 
    of `x` and `y`, respectively.
    """

    # Note: has an alias "cov"

    if len(args) == 0 or len(args) > 2:
        raise ValueError("The function only accepts one or two input "
                         f"arguments, while {len(args)} areguments are given.")
    
    if len(args) == 1:
        return args[0].cov()
    
    # len(args) == 2
    x, y = args
    ax, ay = [em.a2d for em in emaps.complete([x.emap, y.emap])]
    cov2d = ax.T @ ay.conj()
    return cov2d.reshape(x.shape + y.shape)


def cov(*args):
    """Covariance. For a single variable `x` the same as `x.covariance()`.
    
    For two variables `x` and `y` it is defined in an element-wise manner as 
    `<(x-<x>) (y-<y>)^*>`, where `*` is complex conjugation.

    For two arrays `x` and `y` the function returns the array `C` with 
    the number of dimensions equal to the sum of the dimensions of `x` and `y` 
    and the components
    `C[ijk... lmn...] = <(x[ijk..] - <x>) (y[lmn..] - <y>)*>`, 
    where the indices `ijk...` and `lmn...` run over the components 
    of `x` and `y`, respectively.
    """

    # Note: an alias for "covaraince"
    
    return covariance(*args)


# ---------- linear array functions ----------
# These functions apply to numpy arrays without converting them to Normal.


def diagonal(x, offset=0, axis1=0, axis2=1):
    return x.diagonal(offset=offset, axis1=axis1, axis2=axis2)


def sum(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


def cumsum(x, axis=None):
    return x.cumsum(axis=axis)


def moveaxis(x, source, destination):
    if not isinstance(x, Normal):
        return np.moveaxis(x, source, destination)
    return x.moveaxis(source, destination)


def ravel(x, order="C"):
    return x.ravel(order=order)


def reshape(x, newshape, order="C"):
    return x.reshape(newshape, order=order)


def transpose(x, axes=None):
    return x.transpose(axes=axes)


def trace(x, offset=0, axis1=0, axis2=1):
    return x.trace(offset=offset, axis1=axis1, axis2=axis2)


def concatenate(arrays, axis=0):
    return _concatfunc("concatenate", arrays, axis)


def stack(arrays, axis=0):
    return _concatfunc("stack", arrays, axis)


def hstack(arrays):
    return _concatfunc("hstack", arrays)


def vstack(arrays):
    return _concatfunc("vstack", arrays)


def dstack(arrays):
    return _concatfunc("dstack", arrays)


def _concatfunc(name, arrays, *args, **kwargs):
    if not any(isinstance(a, Normal) for a in arrays):
        return getattr(np, name)(arrays, *args, **kwargs)

    arrays = [asnormal(ar) for ar in arrays]
    
    b = getattr(np, name)([x.b for x in arrays], *args, **kwargs)
    em = getattr(emaps, name)([x.emap for x in arrays], *args, **kwargs)
    return Normal(em, b)


def split(x, indices_or_sections, axis=0):   
    if not isinstance(x, Normal):
        return np.split(x, indices_or_sections, axis=axis)
    
    bs = np.split(x.b, indices_or_sections, axis=axis)
    ems = x.emap.split(indices_or_sections, vaxis=axis)
    return [Normal(em, b) for em, b in zip(ems, bs)]


def hsplit(x, indices_or_sections):
    if x.ndim < 1:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    if x.ndim == 1:
        return split(x, indices_or_sections, axis=0)
    return split(x, indices_or_sections, axis=1)


def vsplit(x, indices_or_sections):
    if x.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(x, indices_or_sections, axis=0)


def dsplit(x, indices_or_sections):
    if x.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(x, indices_or_sections, axis=2)
    

def einsum(subs, op1, op2):
    if isinstance(op2, Normal) and isinstance(op1, Normal):
        return einsum(subs, op1.b, op2) + einsum(subs, op1, op2.b)

    if isinstance(op1, Normal) and not isinstance(op2, Normal):
        b = np.einsum(subs, op1.b, op2)
        em = op1.emap.einsum(subs, op2)
        return Normal(em, b)
    
    if isinstance(op2, Normal) and not isinstance(op1, Normal):
        b = np.einsum(subs, op1, op2.b)
        em = op2.emap.einsum(subs, op1, otherfirst=True)
        return Normal(em, b)

    return np.einsum(subs, op1, op2)


def add(op1, op2):
    return op1 + op2


def subtract(op1, op2):
    return op1 - op2


def multiply(op1, op2):
    return op1 * op2


def divide(op1, op2):
    return op1 / op2


def power(op1, op2):
    return op1 ** op2


def matmul(op1, op2):
    return op1 @ op2


def dot(op1, op2):
    return _bilinearfunc("dot", op1, op2)


def inner(op1, op2):
    return _bilinearfunc("inner", op1, op2)


def outer(op1, op2):
    return _bilinearfunc("outer", op1, op2)


def kron(op1, op2):
    return _bilinearfunc("kron", op1, op2)


def tensordot(op1, op2, axes=2):
    return _bilinearfunc("tensordot", op1, op2, axes=axes)


def _bilinearfunc(name, op1, op2, *args, **kwargs):
    if isinstance(op2, Normal) and isinstance(op1, Normal):
        t1 = _bilinearfunc(name, op1.b, op2, *args, **kwargs)
        t2 = _bilinearfunc(name, op1, op2.b, *args, **kwargs)
        return t1 + t2

    if isinstance(op1, Normal) and not isinstance(op2, Normal):
        b = getattr(np, name)(op1.b, op2, *args, **kwargs)
        em = getattr(op1.emap, name)(op2, *args, **kwargs)
        return Normal(em, b)
    
    if isinstance(op2, Normal) and not isinstance(op1, Normal):
        b = getattr(np, name)(op1, op2.b, *args, **kwargs)

        kwargs.update(otherfirst=True)
        em = getattr(op2.emap, name)(op1, *args, **kwargs)
        return Normal(em, b)

    return getattr(np, name)(op1, op2, *args, **kwargs)


# ---------- linear and linearized unary array ufuncs ----------

def linearized_unary(jmpf):
    if not jmpf.__name__.endswith("_jmp"):
        raise ValueError()
    
    f = getattr(np, jmpf.__name__[:-4])

    def flin(x):
        if not isinstance(x, Normal):
            return f(x)
        
        new_b = f(x.b)
        em = jmpf(x.b, new_b, x.emap)
        return Normal(em, new_b)
    
    return flin


# Elementwise Jacobian-matrix products.
def exp_jmp(x, ans, a):     return a * ans
def exp2_jmp(x, ans, a):    return a * (ans * np.log(2.))
def log_jmp(x, ans, a):     return a / x
def log2_jmp(x, ans, a):    return a / (x * np.log(2.))
def log10_jmp(x, ans, a):   return a / (x * np.log(10.))
def sqrt_jmp(x, ans, a):    return a / (2. * ans)
def cbrt_jmp(x, ans, a):    return a / (3. * ans**2)
def sin_jmp(x, ans, a):     return a * np.cos(x)
def cos_jmp(x, ans, a):     return a * (-np.sin(x))
def tan_jmp(x, ans, a):     return a / np.cos(x)**2
def arcsin_jmp(x, ans, a):  return a / np.sqrt(1 - x**2)
def arccos_jmp(x, ans, a):  return a / (-np.sqrt(1 - x**2))
def arctan_jmp(x, ans, a):  return a / (1 + x**2)
def sinh_jmp(x, ans, a):    return a * np.cosh(x)
def cosh_jmp(x, ans, a):    return a * np.sinh(x)
def tanh_jmp(x, ans, a):    return a / np.cosh(x)**2
def arcsinh_jmp(x, ans, a): return a / np.sqrt(x**2 + 1)
def arccosh_jmp(x, ans, a): return a / np.sqrt(x**2 - 1)
def arctanh_jmp(x, ans, a): return a / (1 - x**2)
def conjugate_jmp(x, ans, a): return a.conj()


exp = linearized_unary(exp_jmp)
exp2 = linearized_unary(exp2_jmp)
log = linearized_unary(log_jmp)
log2 = linearized_unary(log2_jmp)
log10 = linearized_unary(log10_jmp)
sqrt = linearized_unary(sqrt_jmp)
cbrt = linearized_unary(cbrt_jmp)
sin = linearized_unary(sin_jmp)
cos = linearized_unary(cos_jmp)
tan = linearized_unary(tan_jmp)
arcsin = linearized_unary(arcsin_jmp)
arccos = linearized_unary(arccos_jmp)
arctan = linearized_unary(arctan_jmp)
sinh = linearized_unary(sinh_jmp)
cosh = linearized_unary(cosh_jmp)
tanh = linearized_unary(tanh_jmp)
arcsinh = linearized_unary(arcsinh_jmp)
arccosh = linearized_unary(arccosh_jmp)
arctanh = linearized_unary(arctanh_jmp)
conjugate = conj = linearized_unary(conjugate_jmp)