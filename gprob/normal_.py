from importlib import import_module
from functools import reduce
from operator import mul

import numpy as np
from numpy.linalg import LinAlgError

from .maps import (LatentMap, complete, lift, match_, concatenate, stack, 
    solve, asolve, call_linearized, fftfunc, fftfunc_n, bilinearfunc, 
    einsum_1, einsum_2, inner_1, inner_2, dot_1, dot_2, outer_1, outer_2, 
    kron_1, kron_2, complete_tensordot_axes, tensordot_1, tensordot_2, a2d)

from .func import condition, logp


class Normal(LatentMap):
    """Array of normal random variables, 
    
    ``x[...] = sum_k a[i...] xi[i] + b[...],``
    
    where ``xi[i]`` are the independent identically-distributed Gaussian 
    random variables: ``xi[i] ~ N(0, 1)`` for all ``i``, and ``...`` is 
    a multi-dimensional index."""

    _mod = import_module(__name__)

    def __repr__(self):
        return print_(self)

    def __or__(self, observations):
        """Conditioning operation."""
        return self.condition(observations)

    def icopy(self):
        """Creates a statistically independent copy of the variable."""
        
        # ``a`` and ``b`` are copied to prevent their in-place modifications 
        # later from affecting the new variable.
        return self.__class__(self.a.copy(), self.b.copy())

    def condition(self, observations, mask=None):
        """Conditioning operation.
        
        Args:
            observations (Normal or dict):
                A single normal variable or a dictionary of observations 
                of the format ``{variable: value, ...}``, where the variables 
                are normal variables, and the values are numeric 
                constants or normal variables. A single normal 
                variable is equavalent to ``{variable: 0}``.
            mask (optional):
                A 2d bool array with generalized upper- or lower- triangular 
                structure that specifies which conditions apply to which 
                variables. ``True`` ``mask[i, j]`` means that 
                the ``i``-th condition applies to the ``j``-th variable, 
                and ``False`` that it does not.
                In the case of many conditions and higher dimensions, 
                the 0th axis of ``mask`` spans over the 0th axis of all 
                conditions, and its 1st axis spans over the 0th axis
                of the conditioned variable.
                The mask has to be generalized upper- or lower- triangular, i.e.
                for some set of indices ``i0[j]`` the mask has to either be 
                ``True`` for all ``i > i0[j]`` and ``False`` for ``i < i0[j]``, 
                or, vice-versa, 
                ``False`` for all ``i > i0[j]`` and ``True`` for ``i < i0[j]``.
        
        Returns:
            Conditional normal variable with the same shape as 
            the original variable.

        Raises:
            ConditionError: 
                If the observations are mutually incompatible,
                or if a mask is given with degenerate observations.
        """
        if isinstance(observations, dict):
            obs = [lift(Normal, k-v) for k, v in observations.items()]
        else:
            obs = [lift(Normal, observations)]

        if not obs:
            return self

        if self.iscomplex:
            # Doubles the dimension preserving the triangular structure.
            self_r = stack(Normal, [self.real, self.imag], axis=-1)
        else:
            self_r = self

        obs_r = []
        for c in obs:
            if c.iscomplex:
                obs_r.extend([c.real, c.imag])
            else:
                obs_r.append(c)

        if mask is None:
            cond = concatenate(Normal, [c.ravel() for c in obs_r])
        else:
            mask = np.asanyarray(mask)

            if mask.ndim != 2:
                raise ValueError("The mask array must have 2 dimensions, "
                                 f"while it has {mask.ndim}.")

            # Concatenates the conditions preserving the element order along 
            # the 0th axis, and expands the mask to the right shape.

            if self.ndim < 1:
                raise ValueError("The variable must have at least one "
                                 "dimension to be conditioned with a mask.")

            if any(c.ndim < 1 for c in obs):
                raise ValueError("All conditions must have at least one "
                                 "dimension to be compatible with masking.")

            obs_r = [c.reshape((c.shape[0], -1)) for c in obs_r]
            cond = concatenate(Normal, obs_r, axis=1).ravel()

            k = sum(c.shape[1] for c in obs_r)
            l = reduce(mul, self_r.shape[1:], 1)
            ms = mask.shape

            mask = np.broadcast_to(mask[:, None, :, None], (ms[0], k, ms[1], l))
            mask = mask.reshape((ms[0] * k, ms[1] * l))

        self_r = self_r.ravel()
        lat, [av_r, ac] = complete([self_r, cond])

        b, a = condition(self_r.b, av_r, cond.b, ac, mask)

        if self.iscomplex:
            # Converting back to complex.
            a = a[:, 0::2] + 1j * a[:, 1::2]
            b = b[0::2] + 1j * b[1::2]

        # Shaping back.
        a = np.reshape(a, (len(lat),) + self.shape)
        b = np.reshape(b, self.shape)

        return Normal(a, b, lat)

    def mean(self):
        """Mean.
        
        Returns:
            An array of the mean values with the same shape as 
            the random variable.
        """
        return self.b

    def var(self):
        """Variance, ``<(x-<x>)(x-<x>)^*>``, where ``*`` denotes 
        complex conjugation, and ``<...>`` is the expectation value of ``...``.
        
        Returns:
            An array of the varaince values with the same shape as 
            the random variable.
        """
      
        return np.real(np.einsum("i..., i... -> ...", self.a, self.a.conj()))

    def cov(self):
        """Covariance, generalizing ``<outer((x-<x>), (x-<x>)^H)>``, 
        where ``H`` denotes conjugate transposition, and ``<...>`` is 
        the expectation value of ``...``.

        Returns:
            An array ``c`` with twice the dimension number as 
            the variable, whose components are 
            ``c[ijk... lmn...] = <(x[ijk..] - <x>)(x[lmn..] - <x>)*>``, 
            where ``ijk...`` and ``lmn...`` are indices that run over 
            the elements of the variable (here ``x``), 
            and ``*`` denotes complex conjugation.

        Examples:
            >>> v = normal(size=(2, 3))
            >>> c = v.cov()
            >>> c.shape
            (2, 3, 2, 3)
            >>> np.all(c.reshape((v.size, v.size)) == np.eye(v.size))
            True
        """

        a = a2d(self)
        cov2d = a.T @ a.conj()
        return cov2d.reshape(self.shape * 2)
    
    def sample(self, n=None):
        """Samples the random variable ``n`` times.
        
        Args:
            n (int or None): 
                The number of samples.
        
        Returns:
            A single sample with the same shape as the variable if ``n`` 
            is ``None``, or an array of samples of the lenght ``n`` if ``n`` 
            is an integer, in which case the total shape of the array is 
            the shape of the variable plus ``(n,)`` prepended as 
            the first dimension.

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
        
        r = np.random.normal(size=nshape + (self.nlat,))
        return (r @ a2d(self) + self.b.ravel()).reshape(nshape + self.shape)
    
    def logp(self, x):
        """Log likelihood of a sample.
    
        Args:
            x (array): 
                A sample value or a sequence of sample values.

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
        a = a2d(self)
        if self.iscomplex:
            # Converts to real by doubling the space size.
            x = np.hstack([x.real, x.imag])
            m = np.hstack([m.real, m.imag])
            a = np.hstack([a.real, a.imag])
        elif np.iscomplexobj(x):
            x = x.astype(x.real.dtype)  # Casts to real with throwing a warning.
        
        cov = a.T @ a 
        return logp(x, m, cov)


def print_(x, extra_attrs=tuple()):
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


def normal(mu=0., sigmasq=1., size=None):
    """Creates a normally-distributed random variable.
    
    Args:
        mu: Scalar mean or array of mean values.
        sigmasq: Scalar variance or matrix covariance.
        size (optional): Integer or tuple of integers specifying the shape 
            of the variable. Only has an effect with scalar mean and variance. 

    Returns:
        Normal: a new random variable.

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
        raise ValueError("Negative eigenvalues in the covariance matrix:\n"
                         f"{eigvals}")
    
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


def cov(x, y):
    """The normal implementation of the covariance between two variables."""

    _, [ax, ay] = complete([x.ravel(), y.ravel()])
    cov2d = ax.T @ ay.conj()
    return cov2d.reshape(x.shape + y.shape)