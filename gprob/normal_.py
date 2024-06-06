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
    _normal_priority_ = 0

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
        return print_normal(self)

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("Scalar arrays have no lengh.")
        return len(self.b)

    def __neg__(self):
        return Normal(-self.emap, -self.b)

    def __add__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.

        other, isnormal = as_numeric_or_normal(other)
        if isnormal:
            return Normal(self.emap + other.emap, self.b + other.b)

        b = self.b + other
        em = self.emap.broadcast_to(b.shape)
        return Normal(em, b)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = as_numeric_or_normal(other)
        if isnormal:
            return Normal(self.emap + (-other.emap), self.b - other.b)
        
        b = self.b - other
        em = self.emap.broadcast_to(b.shape)
        return Normal(em, b)
    
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = as_numeric_or_normal(other)

        if isnormal:
            # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
            # for  x = <x> + dx  and  y = <y> + dy.

            b = self.b * other.b
            em = self.emap * other.b + other.emap * self.b
            return Normal(em, b)
        
        return Normal(self.emap * other, self.b * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = as_numeric_or_normal(other)

        if isnormal:
            # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
            # for  x = <x> + dx  and  y = <y> + dy.

            b = self.b / other.b
            em = self.emap / other.b + other.emap * ((-self.b) / other.b**2)
            return Normal(em, b)
        
        return Normal(self.emap / other, self.b / other)
    
    def __rtruediv__(self, other):
        # Linearized fraction  x/y = <x>/<y> - dy<x>/<y>^2,
        # for  x = <x>  and  y = <y> + dy.

        other, isnormal = as_numeric_or_normal(other)
        if isnormal:
            return other / self
        
        b = other / self.b
        em = self.emap * ((-other) / self.b**2)
        return Normal(em, b)
    
    def __pow__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = as_numeric_or_normal(other)

        if isnormal:
            # x^y = <x>^<y> + dx <y> <x>^(<y>-1) + dy ln(<x>) <x>^<y>
            
            b = self.b ** other.b
            em1 = self.emap * (other.b * self.b ** np.where(other.b, other.b-1, 1.))
            em2 = other.emap * (np.log(np.where(self.b, self.b, 1.)) * b)
            return Normal(em1 + em2, b)
        
        b = self.b ** other
        em = self.emap * (other * (self.b ** np.where(other, other-1, 1.)))
        return Normal(em, b)

    def __rpow__(self, other):
        # x^y = <x>^<y> + dy ln(<x>) <x>^<y>

        other, isnormal = as_numeric_or_normal(other)
        if isnormal:
            return other ** self

        b = other ** self.b
        em = self.emap * (np.log(np.where(other, other, 1.)) * b)
        return Normal(em, b)

    def __matmul__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = as_numeric_or_normal(other)
        if isnormal:
            return self @ other.b + self.b @ other
        
        return Normal(self.emap @ other, self.b @ other)

    def __rmatmul__(self, other):
        other, isnormal = as_numeric_or_normal(other)
        if isnormal:
            return other @ self.b + other.b @ self
        
        return Normal(other @ self.emap, other @ self.b)

    def __getitem__(self, key):
        return Normal(self.emap[key], self.b[key])
    
    def __setitem__(self, key, value):
        value = asnormal(value)

        if not self.b.flags.writeable:
            self.b = self.b.copy()

        self.b[key] = value.b
        self.emap[key] = value.emap

    def __or__(self, observations):
        """Conditioning operation."""
        return self.condition(observations)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""
        return hstack([self, other])  
    
    def __rand__(self, other):
        """Combines two random variables into one vector."""
        return hstack([other, self])
    
    def _is_lower_than(self, other):
        """Checks if `self` has lower operation priority than `other`."""
        
        r = (hasattr(other, "_normal_priority_") 
             and other._normal_priority_ > self._normal_priority_)
        return r

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
    
    def split(self, indices_or_sections, axis=0):
        bs = np.split(self.b, indices_or_sections, axis=axis)
        ems = self.emap.split(indices_or_sections, vaxis=axis)
        return [Normal(em, b) for em, b in zip(ems, bs)]
    
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
    
    @staticmethod
    def concatenate(arrays, axis):
        arrays = [asnormal(ar) for ar in arrays]
        b = np.concatenate([x.b for x in arrays], axis=axis)
        em = emaps.concatenate([x.emap for x in arrays], vaxis=axis)
        return Normal(em, b)
    
    @staticmethod
    def stack(arrays, axis):
        arrays = [asnormal(ar) for ar in arrays]
        b = np.stack([x.b for x in arrays], axis=axis)
        em = emaps.stack([x.emap for x in arrays], vaxis=axis)
        return Normal(em, b)

    @staticmethod
    def bilinearfunc(name, op1, op2, *args, **kwargs):
        op1, isnormal1 = as_numeric_or_normal(op1)
        op2, isnormal2 = as_numeric_or_normal(op2)

        if isnormal2 and isnormal1:
            t1 = _bilinearfunc(name, op1.b, op2, *args, **kwargs)
            t2 = _bilinearfunc(name, op1, op2.b, *args, **kwargs)
            return t1 + t2

        if isnormal1 and not isnormal2:
            b = getattr(np, name)(op1.b, op2, *args, **kwargs)
            em = getattr(op1.emap, name)(op2, *args, **kwargs)
            return Normal(em, b)
        
        if isnormal2 and not isnormal1:
            b = getattr(np, name)(op1, op2.b, *args, **kwargs)

            kwargs.update(otherfirst=True)
            em = getattr(op2.emap, name)(op1, *args, **kwargs)
            return Normal(em, b)

        return getattr(np, name)(op1, op2, *args, **kwargs)

    @staticmethod
    def einsum(subs, op1, op2):
        op1, isnormal1 = as_numeric_or_normal(op1)
        op2, isnormal2 = as_numeric_or_normal(op2)

        if isnormal2 and isnormal1:
            return einsum(subs, op1.b, op2) + einsum(subs, op1, op2.b)

        if isnormal1 and not isnormal2:
            b = np.einsum(subs, op1.b, op2)
            em = op1.emap.einsum(subs, op2)
            return Normal(em, b)
        
        if isnormal2 and not isnormal1:
            b = np.einsum(subs, op1, op2.b)
            em = op2.emap.einsum(subs, op1, otherfirst=True)
            return Normal(em, b)

        return np.einsum(subs, op1, op2)

    @staticmethod
    def fftfunc(name, x, n, axis, norm):
        x = asnormal(x)
        func = getattr(np.fft, name)
        b = func(x.b, n, axis, norm)
        em = x.emap.fftfunc(name, n, axis, norm)
        return Normal(em, b)

    @staticmethod
    def fftfunc_n(name, x, s, axes, norm):
        x = asnormal(x)
        func = getattr(np.fft, name)
        b = func(x.b, s, axes, norm)
        em = x.emap.fftfunc_n(name, s, axes, norm)
        return Normal(em, b)

    # ---------- probability-related methods ----------

    def iid_copy(self):
        """Creates an independent identically distributed copy."""

        # Copies of `a` and `b` are needed for the case if the original array 
        # is in-place modified later. Such modifications should not affect 
        # the new variable.
        return Normal(self.a.copy(), self.b.copy())

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

    def var(self):
        """Variance, `<(x-<x>)(x-<x>)^*>` where `*` is complex conjugation."""

        a = self.emap.a        
        return np.real(np.einsum("i..., i... -> ...", a, a.conj()))

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

        a = self.emap.a2d
        cov2d = a.T @ a.conj()
        return cov2d.reshape(self.shape * 2)
    
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


def print_normal(x, extra_attrs=tuple()):
    csn = x.__class__.__name__

    if x.ndim == 0:
        meanstr = f"{x.mean():.8g}"
        varstr = f"{x.var():.8g}"
        # To make the displays of scalars consistent with the display 
        # of array elements.
    else:
        meanstr = str(x.mean())
        varstr = str(x.var())
    
    d = {"mean": meanstr, "var": varstr}
    d.update({p: str(getattr(x, p)) for p in extra_attrs})

    if all("\n" not in d[k] for k in d):
        s = ", ".join([f"{k}={d[k]}" for k in d])
        return f"{csn}({s})"
    
    padl = max(max([len(k) for k in d]), 6)
    lines = []
    for k in d:
        h = k.rjust(padl, " ") + "="
        addln = d[k].splitlines()
        lines.append(h + addln[0])
        lines.extend(" " * len(h) + l for l in addln[1:])
    
    return "\n".join([f"{csn}(", *lines, ")"])


NUMERIC_ARRAY_KINDS = {"b", "i", "u", "f", "c"}


def as_numeric_or_normal(x):
    """Prepares the operand `x` for an arithmetic operation by converting 
    it to either a numeric array or a normal variable.
    
    Returns:
        Tuple (`numeric_or_normal_x`, `isnormal`)
    """

    if isinstance(x, Normal):
        return x, True
    
    x_ = np.asanyarray(x)
    if x_.dtype.kind not in NUMERIC_ARRAY_KINDS:
        return asnormal(x), True

    return x_, False


def asnormal(x):
    """Converts `x` to a normal variable. If `x` is a normal variable already, 
    returns it unchanged. If `x` is neither a normal nor numeric variable, 
    raises a `TypeError`."""

    if isinstance(x, Normal):
        return x

    b = np.asanyarray(x)
    if b.dtype.kind not in NUMERIC_ARRAY_KINDS:
        if (hasattr(x, "_normal_priority_") 
            and x._normal_priority_ > Normal._normal_priority_):

            raise TypeError(f"The variable {x} cannot be converted to "
                            "a normal variable because it is already "
                            f"of higher priority ({x._normal_priority_} "
                            f"> {Normal._normal_priority_}).")
    
        if b.ndim == 0:
            raise TypeError(f"Variable of type '{x.__class__.__name__}' cannot "
                            "be converted to a normal variable.")
        
        return stack([asnormal(vi) for vi in b])

    em = ElementaryMap(np.zeros((0, *b.shape), dtype=b.dtype))
    return Normal(em, b)


def normal(mu=0., sigmasq=1., size=None):
    """Creates a new normal random variable.
    
    Args:
        mu: Scalar or array mean value.
        sigmasq: Scalar variance or matrix covariance.
        size (optional): Integer or sequence of integers specifying the shape 
            of the variable. Only has an effect with scalar mean and variance. 

    Returns:
        A Normal random variable.
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
        a2dtr = safer_cholesky(sigmasq2d)
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


def safer_cholesky(x):
    ltri = np.linalg.cholesky(x)

    d = np.diagonal(ltri)
    atol = 100 * len(d) * np.finfo(d.dtype).eps * np.max(d**2)
    if (d**2 < atol).any():
        raise LinAlgError("The input matrix seems to be degenerate.")
    
    return ltri


def cov(*args):
    """Covariance. For a single variable `x` returns `x.cov()`.
    
    For two scalar variables `x` and `y` returns the expectation 
    `<(x-<x>) (y-<y>)^*>`, where `*` is complex conjugation.

    For two arrays `x` and `y` returns the array `C` whose number of 
    dimensions is equal to the sum of the dimensions of `x` and `y` and 
    whose components are
    `C[ijk... lmn...] = <(x[ijk..] - <x>) (y[lmn..] - <y>)*>`, 
    where the indices `ijk...` and `lmn...` run over the components 
    of `x` and `y`, respectively.
    """

    if len(args) == 0 or len(args) > 2:
        raise ValueError("The function only accepts one or two input "
                         f"arguments, while {len(args)} areguments are given.")
    
    if len(args) == 1:
        return asnormal(args[0]).cov()
    
    # For the case len(args) == 2.
    x, y = args
    ax, ay = [em.a2d for em in emaps.complete([x.emap, y.emap])]
    cov2d = ax.T @ ay.conj()
    return cov2d.reshape(x.shape + y.shape)


def iid_copy(x):
    """Creates an independent identically distributed copy of `x`."""
    return x.iid_copy()


# ---------- linear array functions ----------


def broadcast_to(x, shape):
    """Broadcasts the normal variable to a new shape."""    
    em = x.emap.broadcast_to(shape)
    b = np.broadcast_to(x.b, shape)
    return Normal(em, b)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return x.diagonal(offset=offset, axis1=axis1, axis2=axis2)


def sum(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


def cumsum(x, axis=None):
    return x.cumsum(axis=axis)


def moveaxis(x, source, destination):
    return x.moveaxis(source, destination)


def ravel(x, order="C"):
    return x.ravel(order=order)


def reshape(x, newshape, order="C"):
    return x.reshape(newshape, order=order)


def transpose(x, axes=None):
    return x.transpose(axes=axes)


def trace(x, offset=0, axis1=0, axis2=1):
    return x.trace(offset=offset, axis1=axis1, axis2=axis2)


def get_highest_class(seq):
    """Returns the class of the highest-priority object in the sequence `seq`
    according to `_normal_priority_`, defaulting to `Normal`."""

    obj = max(seq, default=0, key=lambda a: getattr(a, "_normal_priority_", 
                                                    Normal._normal_priority_-1))
    cls = obj.__class__
    if not hasattr(cls, "_normal_priority_"):
        return Normal 
    
    return cls


def concatenate(arrays, axis=0):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to concatenate.")
    cls = get_highest_class(arrays)
    return cls.concatenate(arrays, axis)


def stack(arrays, axis=0):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    cls = get_highest_class(arrays)
    return cls.stack(arrays, axis)


def _as_array_seq(arrays_or_scalars):
    return [x if hasattr(x, "ndim") else np.array(x) for x in arrays_or_scalars]
    

def hstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    arrays = _as_array_seq(arrays)
    arrays = [a.ravel() if a.ndim == 0 else a for a in arrays]
    if arrays[0].ndim == 1:
        return concatenate(arrays, axis=0)
    
    return concatenate(arrays, axis=1)
    

def vstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    arrays = _as_array_seq(arrays)
    if arrays[0].ndim <= 1:
        arrays = [a.reshape((1, -1)) for a in arrays]
    
    return concatenate(arrays, axis=0)


def dstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    arrays = _as_array_seq(arrays)
    if arrays[0].ndim <= 1:
        arrays = [a.reshape((1, -1, 1)) for a in arrays]
    elif arrays[0].ndim == 2:
        arrays = [a.reshape((*a.shape, 1)) for a in arrays]
    
    return concatenate(arrays, axis=2)


def split(x, indices_or_sections, axis=0):   
    return x.split(indices_or_sections=indices_or_sections, axis=axis)


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
    cls = get_highest_class([op1, op2])
    return cls.einsum(subs, op1, op2)


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
    cls = get_highest_class([op1, op2])
    return cls.bilinearfunc(name, op1, op2, *args, **kwargs)


# ---------- linear and linearized unary array ufuncs ----------

def linearized_unary(jmpf):
    if not jmpf.__name__.endswith("_jmp"):
        raise ValueError()
    
    fnm = jmpf.__name__[:-4]
    f = getattr(np, fnm)

    def flin(x):
        x, isnormal = as_numeric_or_normal(x)

        if not isnormal:
            return f(x)
        
        new_b = f(x.b)
        em = jmpf(x.b, new_b, x.emap)
        return Normal(em, new_b)
    
    flin.__name__ = fnm
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