from functools import reduce
from operator import mul
from builtins import sum as sum_

import numpy as np
from numpy.linalg import LinAlgError

from . import emaps
from .emaps import ElementaryMap
from . import func


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

        other, isnormal = _as_numeric_or_normal(other)
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
        
        other, isnormal = _as_numeric_or_normal(other)
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
        
        other, isnormal = _as_numeric_or_normal(other)

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
        
        other, isnormal = _as_numeric_or_normal(other)

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

        other, isnormal = _as_numeric_or_normal(other)
        if isnormal:
            return other / self
        
        b = other / self.b
        em = self.emap * ((-other) / self.b**2)
        return Normal(em, b)
    
    def __pow__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = _as_numeric_or_normal(other)

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

        other, isnormal = _as_numeric_or_normal(other)
        if isnormal:
            return other ** self

        b = other ** self.b
        em = self.emap * (np.log(np.where(other, other, 1.)) * b)
        return Normal(em, b)

    def __matmul__(self, other):
        if self._is_lower_than(other):
            return NotImplemented  # The operation must be handled by `other`.
        
        other, isnormal = _as_numeric_or_normal(other)
        if isnormal:
            b = self.b @ other.b
            em = self.emap @ other.b
            em += self.b @ other.emap
            return Normal(em, b)
        
        return Normal(self.emap @ other, self.b @ other)

    def __rmatmul__(self, other):
        other, isnormal = _as_numeric_or_normal(other)
        if isnormal:
            b = other.b @ self.b
            em = other.b @ self.emap
            em += other.emap @ self.b
            return Normal(em, b)
        
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
        return stack([self, other])  
    
    def __rand__(self, other):
        """Combines two random variables into one vector."""
        return stack([other, self])
    
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

        newshape = b.shape
        # This replaces possible -1 in newshape with an explicit value,
        # which is not easy for the emap's method to calculate in the case 
        # if the normal variable is deterministic, meaning that a.size == 0.

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

    # ---------- probability-related methods ----------

    def iid_copy(self):
        """Creates an independent identically distributed copy 
        of the varaible."""

        # Copies of `a` and `b` are taken becase if the original variable 
        # is in-place modified later, those modifications should not affect 
        # the new variable.
        return Normal(self.a.copy(), self.b.copy())

    def condition(self, observations, mask=None):
        """Conditioning operation.
        
        Args:
            observations (Normal or dict):
                A single normal variable or a dictionary of observations
                of the format 
                {`variable`: `value`, ...}, where `variable`s are normal 
                variables, and `value`s can be numerical constants or 
                normal variables. Specifying a single normal `variable` 
                is equavalent to specifying {`variable`: `0`}.
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
            ConditionError: 
                If the observations are mutually incompatible,
                or if a mask is given with degenerate observations.
        """
        if isinstance(observations, dict):
            obs = [asnormal(k-v) for k, v in observations.items()]
        else:
            obs = [asnormal(observations)]

        if not obs:
            return self

        if self.iscomplex:
            # Doubles the dimension preserving the triangular structure.
            self_r = stack([self.real, self.imag], axis=-1)
        else:
            self_r = self

        obs_r = []
        for c in obs:
            if c.iscomplex:
                obs_r.extend([c.real, c.imag])
            else:
                obs_r.append(c)

        if mask is None:
            cond = concatenate([c.ravel() for c in obs_r])
        else:
            # Concatenates the conditions preserving the element order along 
            # the 0th axis, and expands the mask to the right shape.

            if self.ndim < 1:
                raise ValueError("The variable must have at least one "
                                 "dimension to be conditioned with a mask.")

            if any(c.ndim < 1 for c in obs):
                raise ValueError("All conditions must have at least one "
                                 "dimension to be compatible with masking.")

            obs_r = [c.reshape((c.shape[0], -1)) for c in obs_r]
            cond = concatenate(obs_r, axis=1).ravel()

            k = sum_(c.shape[1] for c in obs_r)
            l = reduce(mul, self_r.shape[1:], 1)
            ms = mask.shape

            mask = np.broadcast_to(mask[:, None, :, None], (ms[0], k, ms[1], l))
            mask = mask.reshape((ms[0] * k, ms[1] * l))

        emv, emc = emaps.complete((self_r.emap, cond.emap))

        a = emv.a2d
        m = self_r.b.ravel()
        ac = emc.a
        mc = cond.b

        new_b, new_a = func.condition(m, a, mc, ac, mask)

        if self.iscomplex:
            # Converting back to complex.
            new_a = new_a[:, 0::2] + 1j * new_a[:, 1::2]
            new_b = new_b[0::2] + 1j * new_b[1::2]

        # Shaping back.
        new_a = np.reshape(new_a, (new_a.shape[0],) + self.shape)
        new_b = np.reshape(new_b, self.b.shape)

        return Normal(ElementaryMap(new_a, emv.elem), new_b)

    def mean(self):
        """Mean.
        
        Returns:
            An array of the mean values with the same shape as 
            the random variable.
        """
        return self.b

    def var(self):
        """Variance, `<(x-<x>)(x-<x>)^*>`, where `*` denotes 
        complex conjugation, and `<...>` is the expectation value of `...`.
        
        Returns:
            An array of the varaince values with the same shape as 
            the random variable.
        """

        a = self.emap.a        
        return np.real(np.einsum("i..., i... -> ...", a, a.conj()))

    def cov(self):
        """Covariance, generalizing `<outer((x-<x>), (x-<x>)^H)>`, 
        where `H` denotes conjugate transposition, and `<...>` is 
        the expectation value of `...`.

        Returns:
            An array `c` with twice the dimension number as 
            the variable, whose components are 
            `c[ijk... lmn...] = <(x[ijk..] - <x>)(x[lmn..] - <x>)*>`, 
            where `ijk...` and `lmn...` are indices that run over 
            the elements of the variable (here `x`), 
            and `*` denotes complex conjugation.

        Examples:
            >>> v = normal(size=(2, 3))
            >>> c = v.cov()
            >>> c.shape
            (2, 3, 2, 3)
            >>> np.all(c.reshape((v.size, v.size)) == np.eye(v.size))
            True
        """

        a = self.emap.a2d
        cov2d = a.T @ a.conj()
        return cov2d.reshape(self.shape * 2)
    
    def sample(self, n=None):
        """Samples the random variable `n` times.
        
        Args:
            n: An integer number of samples or None.
        
        Returns:
            A single sample with the same shape as the varaible if `n` is None, 
            or an array of samples of the lenght `n` if `n` is an integer,
            in which case the total shape of the array is the shape of 
            the varaible plus (n,) prepended as the first dimension.

        Examples:
            >>> v.shape
            (2, 3)
            >>> v.sample()
            array([[-0.33993954, -0.26758247, -0.33927517],
                   [-0.36414751,  0.76830802,  0.0997399 ]])
            >>> v.sample(2)
            array([[[-1.78808198,  1.08481027,  0.40414722],
                    [ 0.95298205, -0.42652839,  0.62417706]],

                   [[-0.81295799,  1.76126207, -0.36532098],
                    [-0.22637276, -0.67915003, -1.55995937]]])
            >>> v.sample(5).shape
            (5, 2, 3)
        """

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
        validate_logp_samples(self, x)

        # Flattens the sample values.
        x = x.reshape(x.shape[0: (x.ndim - self.ndim)] + (self.size,))

        m = self.b.ravel()
        a = self.emap.a2d
        if self.iscomplex:
            # Converts to real by doubling the space size.
            x = np.hstack([x.real, x.imag])
            m = np.hstack([m.real, m.imag])
            a = np.hstack([a.real, a.imag])
        elif np.iscomplexobj(x):
            x = x.astype(x.real.dtype)  # Casts to real with throwing a warning.
        
        cov = a.T @ a 
        return func.logp(x, m, cov)


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


def validate_logp_samples(v, x):
    if x.shape != v.shape and x.shape[1:] != v.shape:
        if x.ndim > v.ndim:
            s = f"The the shape of the array of samples {x.shape}"
        else:
            s = f"The the sample shape {x.shape}"

        raise ValueError(f"{s} is not consistent "
                            f"with the variable shape {v.shape}.")


NUMERIC_ARRAY_KINDS = {"b", "i", "u", "f", "c"}


def _as_numeric_or_normal(x):
    """Prepares the operand `x` for an arithmetic operation by converting 
    it to either a numeric array or a normal variable.

    Args:
        x: 
            Normal variable, or a variable convertible to a numeric array.
            In particular, it should be ensured externally that `x` 
            is not a higher-type random variable.  
    
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
    elif hasattr(x, "_normal_priority_"):
        if x._normal_priority_ > Normal._normal_priority_:
            raise TypeError(f"The variable of type {x.__class__.__name__} "
                            "cannot be converted to a normal variable because "
                            f"it is of higher type ({x._normal_priority_} "
                            f"> {Normal._normal_priority_}).")
        else:
            # For possible future extensions.
            raise TypeError("Unknown normal subtype.")

    b = np.asanyarray(x)
    if b.dtype.kind not in NUMERIC_ARRAY_KINDS:
        if b.ndim == 0:
            raise TypeError(f"The variable of type {x.__class__.__name__} "
                            "cannot be converted to a normal variable.")
        
        return stack([asnormal(vi) for vi in x])

    em = ElementaryMap(np.zeros((0, *b.shape), dtype=b.dtype))
    return Normal(em, b)


def normal(mu=0., sigmasq=1., size=None):
    """Creates a new normal random variable.
    
    Args:
        mu: Scalar mean value or array of mean values.
        sigmasq: Scalar variance or matrix covariance.
        size (optional): Integer or sequence of integers specifying the shape 
            of the variable. Only has an effect with scalar mean and variance. 

    Returns:
        A Normal random variable.

    Examples:
        >>> v = normal(1, 3, size=2)
        >>> v.mean()
        array([1, 1])
        >>> v.cov()
        array([[3., 0.],
               [0., 3.]])

        >>> v = normal([0.5, 0.1], [[2, 1], [1, 2]])
        >>> v.mean()
        array([0.5, 0.1])
        >>> v.cov()
        array([[2., 1.],
               [1., 2.]])
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
    """The normal implementation of the covariance. The function 
    expects `args` to have strictly one or two elements."""

    args = [asnormal(arg) for arg in args]
    
    if len(args) == 1:
        return args[0].cov()
    
    # The remaining case is len(args) == 2.
    x, y = args
    ax, ay = [em.a2d for em in emaps.complete([x.emap, y.emap])]
    cov2d = ax.T @ ay.conj()
    return cov2d.reshape(x.shape + y.shape)


# ---------- array functions ----------


def broadcast_to(x, shape):
    """Broadcasts the normal variable to a new shape.""" 
    x = asnormal(x)   
    em = x.emap.broadcast_to(shape)
    b = np.broadcast_to(x.b, shape)
    return Normal(em, b)


def concatenate(arrays, axis=0):
    arrays = [asnormal(ar) for ar in arrays]
    b = np.concatenate([x.b for x in arrays], axis=axis)
    em = emaps.concatenate([x.emap for x in arrays], vaxis=axis)
    return Normal(em, b)


def stack(arrays, axis=0):
    arrays = [asnormal(ar) for ar in arrays]
    b = np.stack([x.b for x in arrays], axis=axis)
    em = emaps.stack([x.emap for x in arrays], vaxis=axis)
    return Normal(em, b)


def bilinearfunc(name, op1, op2, args=tuple(), pargs=tuple()):
    op1, isnormal1 = _as_numeric_or_normal(op1)
    op2, isnormal2 = _as_numeric_or_normal(op2)

    if isnormal2 and isnormal1:
        b = getattr(np, name)(*pargs, op1.b, op2.b, *args)
        em = getattr(op1.emap, name)(*pargs, op2.b, *args)
        em += getattr(op2.emap, name)(*pargs, op1.b, *args, otherfirst=True)
        return Normal(em, b)

    if isnormal1 and not isnormal2:
        b = getattr(np, name)(*pargs, op1.b, op2, *args)
        em = getattr(op1.emap, name)(*pargs, op2, *args)
        return Normal(em, b)
    
    if isnormal2 and not isnormal1:
        b = getattr(np, name)(*pargs, op1, op2.b, *args)
        em = getattr(op2.emap, name)(*pargs, op1, *args, otherfirst=True)
        return Normal(em, b)

    return getattr(np, name)(*pargs, op1, op2, *args)


def dot(op1, op2): 
    return bilinearfunc("dot", op1, op2)


def inner(op1, op2): 
    return bilinearfunc("inner", op1, op2)


def outer(op1, op2):
    return bilinearfunc("outer", op1, op2)


def kron(op1, op2):
    return bilinearfunc("kron", op1, op2)


def tensordot(op1, op2, axes=2):
    return bilinearfunc("tensordot", op1, op2, [axes])


def einsum(subs, op1, op2):
    return bilinearfunc("einsum", op1, op2, pargs=[subs])


def fftfunc(name, x, n, axis, norm):
    x = asnormal(x)
    func = getattr(np.fft, name)
    b = func(x.b, n, axis, norm)
    em = x.emap.fftfunc(name, n, axis, norm)
    return Normal(em, b)


def fftfunc_n(name, x, s, axes, norm):
    x = asnormal(x)
    func = getattr(np.fft, name)
    b = func(x.b, s, axes, norm)
    em = x.emap.fftfunc_n(name, s, axes, norm)
    return Normal(em, b)


def call_linearized(x, func, jmpfunc):
    x = asnormal(x)
    new_b = func(x.b)
    em = jmpfunc(x.b, new_b, x.emap)
    return Normal(em, new_b)