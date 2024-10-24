import numpy as np

import jax
from jax import numpy as jnp

from gprob.normal_ import Normal, complete, validate_logp_samples
from gprob.func import logp, dlogp, fisher


def jmp(fun, primals, tangents):
    """Forward mode jacobain-matrix product for ``fun``. Spans over the 0-th 
    dimension of each of the arrays in ``tangents``, and stacks the results 
    the along the 0-th dimension of the output.
    
    Args:
        primals: A list or tuple of positional arguments to ``fun`` at which its 
            Jacobian should be evaluated.
        tangents: A list or tuple of arrays of positional tangent vectors with
            the same structure as primals, and each array shape being augmented 
            by one outer-most dimension compared to primals.

    Returns:
        Jacobian-matrix product.
    """

    # The function operates on transposed input matrices and calculates
    # m_out = J @ m.T, because vmap works the fastest along the inner-most axis
    # for C-ordered arrays.
    #
    # The function can handle multiple arguments, but otherwise the same as in
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

    jvp = lambda t: jax.jvp(fun, primals, t)[1]
    return jax.vmap(jvp, in_axes=0, out_axes=0)(tangents)


def pnormal(f, input_vs, jit=True):
    """Creates a parametric normal random variable from a function with 
    the signature ``f(p, vs) -> u``, where ``p`` is a 1D array of parameters, 
    ``vs`` is an input array or sequence of arrays whose shapes are consistent 
    with the input random variables, and ``u`` is an output array.
    
    Args:
        f: generating function.
        input_vs: input random variables - a single normal variable or a 
            sequence of normal variables.
        jit: if jax.jit should be applied to the intermediate functions.

    Returns:
        A parametric normal variable corresponding to the random function 
        ``lambda p: f(p, input_vs)``.
    """
    # ``p``` is limited to 1D with the scipy.minimize signature in mind

    if isinstance(input_vs, (list, tuple)):
        inbs = tuple(force_float(v.b) for v in input_vs)
        lat, inas = complete(input_vs)

        afun = lambda p: jmp(lambda *v: f(p, v), inbs, tuple(inas))
        bfun = lambda p: f(p, inbs)
    elif isinstance(input_vs, Normal):
        lat = input_vs.lat
        afun = lambda p: jmp(lambda v: f(p, v), (force_float(input_vs.b),),
                             (input_vs.a,))
        bfun = lambda p: f(p, input_vs.b)
    else:
        raise ValueError("vs must be a normal variable or a sequence of normal "
                         f"variables, while it is of type '{type(input_vs)}'.")

    dafun = lambda p: jmp(afun, (jnp.array(p),), (jnp.eye(len(p)),))
    dbfun = lambda p: jmp(bfun, (jnp.array(p),), (jnp.eye(len(p)),))

    # Note: above, the conversion of ``p`` to jnp array is necessary when  
    # calculating the derivatives over ``p``, as the values of ``p`` will 
    # be input by the user, and the behavior of jax.jvp is sensitive to 
    # the data type. In contrast, in ``afun`` the inputs to jax.jvp are 
    # already always arrays.

    if jit:
        afun = jax.jit(afun)
        dafun = jax.jit(dafun)
        bfun = jax.jit(bfun)
        dbfun = jax.jit(dbfun)

    return ParametricNormal(afun, bfun, dafun, dbfun, lat)
    

class ParametricNormal:
    """Parametric normal random variable produced by a linearized function."""

    __slots__ = ("_afun", "_bfun", "_dafun", "_dbfun", "lat")

    def __init__(self, afun, bfun, dafun, dbfun, lat):
        # ijk... is the array index of the variable, nlat is the number 
        # of the latent variables, np is the number of the parameters.

        self._afun = afun       # output shape:    nlat x ijk... 
        self._bfun = bfun       # output shape:    ijk...
        self._dafun = dafun     # output shape:    np x nlat x ijk... 
        self._dbfun = dbfun     # output shape:    np x ijk...
        self.lat = lat

    def __call__(self, p):
        return Normal(self.a(p), self.mean(p), self.lat)

    # Note: the private functions produce jax arrays, which have to be 
    # explicitly converted to regular numpy arrays for further calculations.

    def a(self, p):
        return np.array(self._afun(p))

    def da(self, p):  
        return np.array(self._dafun(p))
    
    def mean(self, p):
        return np.array(self._bfun(p))
    
    def dmean(self, p):
        return np.array(self._dbfun(p))
    
    def cov(self, p):
        return self(p).cov()
    
    def var(self, p):
        return self(p).var()

    def _d01(self, p):
        """Mean vector and covariance matrix with their derivatives.

        Args:
            p: sequence of parameter values.
        
        Returns:
            (m, dm/dp, cov, dcov/dp), all flattened and converted to real.    
        """

        m = self.mean(p).ravel()
        dm = self.dmean(p).reshape((-1, len(m)))

        a = self.a(p)
        a = a.reshape((a.shape[0], -1))

        da = self.da(p)
        da = da.reshape((-1,) + a.shape)

        if (np.iscomplexobj(a) or np.iscomplexobj(da) 
            or np.iscomplexobj(m) or np.iscomplexobj(dm)):
            
            # Converts to real by doubling the space size.
            m = np.hstack([m.real, m.imag])
            dm = np.hstack([dm.real, dm.imag])
            a = np.hstack([a.real, a.imag])
            da = np.concatenate([da.real, da.imag], axis=-1)

            iscomplex = True
        else:
            iscomplex = False

        cov = a.T @ a
        prod1 = a.T @ da
        dcov = prod1 + prod1.transpose(0, 2, 1)

        return m, dm, cov, dcov, iscomplex

    def logp(self, p, x):
        m = self.mean(p)
        a = self.a(p)

        x = np.asanyarray(x)
        validate_logp_samples(m, x)

        # Flattens the sample values.
        x = x.reshape(x.shape[0: (x.ndim - m.ndim)] + (m.size,))

        m = m.ravel()
        a = a.reshape((a.shape[0], -1))

        if np.iscomplexobj(a) or np.iscomplexobj(m):
            # Converts to real by doubling the space size.
            x = np.hstack([x.real, x.imag])
            m = np.hstack([m.real, m.imag])
            a = np.hstack([a.real, a.imag])
        elif np.iscomplexobj(x):
            x = x.astype(x.real.dtype)  # Casts to real with throwing a warning.
        
        cov = a.T @ a 
        return logp(x, m, cov)

    def dlogp(self, p, x):
        """The gradient of the log probability density."""

        x = np.asanyarray(x).ravel()
        m, dm, cov, dcov, iscomplex = self._d01(p)
        if iscomplex:
            x = np.hstack([x.real, x.imag])

        return dlogp(x, m, cov, dm, dcov)
    
    def fisher(self, p):
        """Fisher information matrix."""

        _, dm, cov, dcov, _ = self._d01(p)
        return fisher(cov, dm, dcov)
    
    def natdlogp(self, p, x):
        """Natural gradient."""

        x = np.asanyarray(x).ravel()
        m, dm, cov, dcov, iscomplex = self._d01(p)
        if iscomplex:
            x = np.hstack([x.real, x.imag])

        g = dlogp(x, m, cov, dm, dcov)
        fimat = fisher(cov, dm, dcov)
        return np.linalg.solve(fimat, g)
    

def force_float(x):
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64)
    return x