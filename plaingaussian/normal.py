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

    def __repr__(self):
        def rpad(sl):
            max_len = max(len(l) for l in sl)
            return [l.ljust(max_len, " ") for l in sl] # TODO: fix for the case when one array starts wrapping and the other does not
        
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
        return Normal(self.emap @ other, self.b @ other)

    def __rmatmul__(self, other):
        if isinstance(other, Normal):
            raise NotImplementedError
        return Normal(other @ self.emap, other @ self.b)

    def __getitem__(self, key):
        return Normal(self.emap[key], self.b[key])

    def __or__(self, observations: dict):
        """Conditioning operation."""
        return self.condition(observations)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""

        other = asnormal(other)

        if self.ndim > 1 or other.ndim > 1:
            raise ValueError("& is only applicable to 0- and 1-d arrays.")
        
        b = np.concatenate([self.b.ravel(), other.b.ravel()], axis=0)
        em = emaps.join([self.emap, other.emap])
        return Normal(em, b)  
    
    def __rand__(self, other):
        """Combines two random variables into one vector.""" # TODO: add test cases for this

        other = asnormal(other)  # other is always a constant

        if self.ndim > 1 or other.ndim > 1:
            raise ValueError("& is only applicable to 0- and 1-d arrays.")
        
        b = np.concatenate([other.b.ravel(), self.b.ravel()], axis=0)
        em = emaps.join([other.emap, self.emap])
        return Normal(em, b)

    # ---------- array methods ----------

    def conj(self):
        return Normal(self.emap.conj(), self.b.conj())
    
    def flatten(self, order="C"):
        b = self.b.flatten(order=order)
        em = self.emap.flatten(order=order)
        return Normal(em, b)
    
    def cumsum(self, axis=None, dtype=None):    
        b = self.b.cumsum(axis, dtype=dtype)
        em = self.emap.cumsum(axis, dtype=dtype)
        return Normal(em, b)
    
    def ravel(self):
        # Order is always "C"
        b = self.b.ravel()
        em = self.emap.ravel()
        return Normal(em, b)
    
    def reshape(self, newshape, order="C"):
        b = self.b.reshape(newshape, order=order)
        em = self.emap.reshape(newshape, order=order)
        return Normal(em, b)
    
    def sum(self, axis=None, dtype=None, keepdims=False):
        # "where" is absent because its broadcasting is not implemented.
        # "initial" is also not implemented.
        b = self.b.sum(axis, dtype=dtype, keepdims=keepdims)
        em = self.emap.sum(axis, dtype=dtype, keepdims=keepdims)
        return Normal(em, b)
    
    def transpose(x, axes=None):
        b = x.b.transpose(axes)
        em = x.emap.transpose(axes)
        return Normal(em, b)

    # ---------- probability-related methods ----------

    def condition(self, observations: dict, mask=None):
        """Conditioning operation.
        
        Args:
            observations: 
                A dictionary of observations {`variable`: `value`, ...}, where 
                `variable`s are normal variables, and `value`s can be constants 
                or normal variables.
            mask (optional): 
                A 2d bool array, in which `mask[i, j] == False` means that 
                the `i`-th condition should not affect the `j`-th variable.
                The 0th axis of `mask` spans over the conditions, and its 1st 
                axis spans over the variable. To variables and conditions whose 
                numbers of dimensions are greater than one the mask applies  
                along their 0-th axes.
                The mask needs to be generalized upper- or lower- triangular, 
                meaning that there needs to be a set of indices `i0[j]` such 
                that either `mask[i, j] == True` for `i > i0[j]` and 
                `mask[i, j] == False` for `i < i0[j]`, or `mask[i, j] == True` 
                for `i < i0[j]` and `mask[i, j] == False` for `i > i0[j]`.
        
        Returns:
            Conditional normal variable.

        Raises:
            ConditionError if the observations are mutually incompatible,
            or if a mask is given with degenerate observations.
        """

        if mask is None:
            cond = concatenate([(k-v).ravel() for k, v in observations.items()])
        
        else:
            # Concatenates preserving the order along the 0th axis.

            if self.ndim < 1:
                raise ValueError("The variable must have at least one "
                                 "dimension to be conditioned with a mask.")

            obs = [k-v for k, v in observations.items()]
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
        new_b, new_a = condition(self.b.ravel(), emv.a2d, cond.b, emc.a, mask)

        # Shaping back
        new_a = np.reshape(new_a, emv.a.shape)
        new_b = np.reshape(new_b, self.b.shape)

        return Normal(ElementaryMap(new_a, emv.elem), new_b)

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
        a = self.emap.a2d

        if np.iscomplexobj(a):
            return a.T.conj() @ a

        return a.T @ a
    
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
            if x.ndim == self.ndim:
                x = x.reshape((x.size,))
            else:
                x = x.reshape((x.shape[0], -1)) 
        return logp(x, self.b.ravel(), self.cov())


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


# ---------- linear array functions ----------

# These functions apply to numpy arrays without forcing their convertion 
# to Normal. At the same time, they omit some arguments supported by 
# the corresponding numpy functions.

# TODO: concatenate functions actually do force conversion

def sum(x, axis=None, dtype=None, keepdims=False):
    return x.sum(axis=axis, dtype=dtype, keepdims=keepdims)


def cumsum(x, axis=None, dtype=None):
    return x.cumsum(axis=axis, dtype=dtype)


def ravel(x):
    return x.ravel()


def reshape(x, newshape, order="C"):
    return x.reshape(newshape, order=order)


def transpose(x, axes=None):
    return x.transpose(axes=axes)


def concatenate(arrays, axis=0, dtype=None):
    return _concatfunc("concatenate", arrays, axis, dtype=dtype)


def stack(arrays, axis=0, dtype=None):
    return _concatfunc("stack", arrays, axis, dtype=dtype)


def hstack(arrays, dtype=None):
    return _concatfunc("hstack", arrays, dtype=dtype)


def vstack(arrays, dtype=None):
    return _concatfunc("vstack", arrays, dtype=dtype)


def dstack(arrays, dtype=None):
    return _concatfunc("dstack", arrays, dtype=dtype)


# Encompasses concatenate, stack, hstack, vstack, dstack
def _concatfunc(name, arrays, *args, **kwargs):
    arrays = [asnormal(ar) for ar in arrays]
    
    if len(arrays) == 0:
        raise ValueError("Need at least one array.")
    elif len(arrays) == 1:
        return arrays[0]
    
    b = getattr(np, name)([x.b for x in arrays], *args, **kwargs)
    em = getattr(emaps, name)([x.emap for x in arrays], *args, **kwargs)

    return Normal(em, b)


# TODO: split family: split, hsplit, vsplit, dsplit

# TODO: linear algebra family: dot, matmul, einsum, inner, outer, kron

def einsum(subs, op1, op2):
    if isinstance(op2, Normal) and isinstance(op1, Normal):
        raise NotImplementedError("Einsums between two normal variables are not implemented.")

    if isinstance(op1, Normal) and not isinstance(op2, Normal):
        b = np.einsum(subs, op1.b, op2)
        em = op1.emap.einsum(subs, op2)
        return Normal(em, b)
    
    if isinstance(op2, Normal) and not isinstance(op1, Normal):
        b = np.einsum(subs, op1, op2.b)
        em = op2.emap.einsum(subs, op1, otherfirst=True)
        return Normal(em, b)

    return np.einsum(subs, op1, op2)