import numpy as np

import jax
from jax import numpy as jnp

from .normal import Normal
from .elementary import u_complete_maps
from .func import logp, dlogp, fisher


def jmp(f, primals, tangents):
    """Jacobain-matrix product."""

    # Note: operates on transposed matrices and calculates m_out = J @ m.T,
    # because vmap works the fastest along the inner-most axis,
    #
    # The function can handle multiple arguments, but otherwise the same as in
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

    jvp = lambda t: jax.jvp(f, primals, t)[1]  # TODO: specify the axes explicitly
    return jax.vmap(jvp)(tangents)


def pnormal(f, input_vs, jit=True):
    """Creates a parametric normal variable from a function with signature 
    `f(p, vs) -> u`, where `p` is a 1D array of parameters, `vs` is an input 
    array or sequence of arrays, and `u` is an output array.
    
    Args:
        f: the generating function.
        input_vs: input random variables - a normal variable or a sequence of 
        normal variables.
        jit: if jax.jit should be used to speed up the intermediate functions.

    Returns:
        A parametric normal variable representing the random function 
        `lambda p: f(p, input_vs)`.
    """
    # `p` is limited to 1D with the scipy minimize signature in mind

    if isinstance(input_vs, (list, tuple)):
        inbs = tuple(v.b.astype('float') for v in input_vs)                   #TODO: fix the type handling
        inas, iids = u_complete_maps([(v.a, v.iids) for v in input_vs])
        afun = lambda p: jmp(lambda *v: f(p, v), inbs, inas)
        bfun = lambda p: f(p, inbs)
    elif isinstance(input_vs, Normal):
        iids = input_vs.iids
        afun = lambda p: jmp(lambda v: f(p, v), (input_vs.b,), (input_vs.a,))
        bfun = lambda p: f(p, input_vs.b)
    else:
        raise ValueError("vs must be a normal variable or a sequence of normal "
                         f"variables, while it is of type '{type(input_vs)}'.")

    dafun = lambda p: jmp(afun, (jnp.array(p),), (jnp.eye(len(p)),))
    dbfun = lambda p: jmp(bfun, (jnp.array(p),), (jnp.eye(len(p)),))

    # Note: above, the conversion of `p` to jnp array is necessary when  
    # calculating the derivatives over `p`, as the values of `p` will be input 
    # by the user, and the behavior of jax.jvp is sensitive to the data type. 
    # In contrast, in `afun` the inputs to jax.jvp are already always arrays.

    if jit:
        afun = jax.jit(afun)
        dafun = jax.jit(dafun)
        bfun = jax.jit(bfun)
        dbfun = jax.jit(dbfun)

    return ParametricNormal(afun, bfun, dafun, dbfun, iids)
    

class ParametricNormal:
    """Parametric normal random variable produced by a linearized function."""

    # TODO: annotate the matrix shapes

    __slots__ = ("_afun", "_bfun", "_dafun", "_dbfun", "iids")

    def __init__(self, afun, bfun, dafun, dbfun, iids):
        self._afun = afun  # TODO: signature.
        self._bfun = bfun
        self._dafun = dafun
        self._dbfun = dbfun
        self.iids = iids

    def __call__(self, p):
        return Normal(self.a(p), self.mean(p), self.iids)

    # Note: the underscored functions produce jax arrays, which have to be 
    # explicitly converted to regular numpy arrays.

    def a(self, p):
        return np.array(self._afun(p))  # nf x nrv

    def da(self, p):  
        return np.array(self._dafun(p))  # np x nf x niids
    
    def mean(self, p):
        return np.array(self._bfun(p))
    
    def dmean(self, p):
        return np.array(self._dbfun(p))  # np x nf
    
    def cov(self, p):
        a = self.a(p)
        return a.T @ a

    def d01cov(self, p):
        """Covariance matrix with its derivative.

        Args:
            p: An array or sequence of parameter values.
        
        Returns:
            (covariance, dcovariance/dp)    
        """
        
        a = self.a(p)
        cov = a.T @ a
        da = self.da(p)
        prod1 = a.T @ da
        dcov = prod1 + prod1.transpose(0, 2, 1)

        return cov, dcov

    def logp(self, p, x):
        m = self.mean(p)
        cov = self.cov(p)
        return logp(x, m, cov)

    def dlogp(self, p, x):
        """The gradient of the log probability density."""

        m = self.mean(p)
        dm = self.dmean(p)
        cov, dcov = self.d01cov(p)

        return dlogp(x, m, cov, dm, dcov)
    
    def fisher(self, p):
        """Fisher information matrix."""

        dm = self.dmean(p)
        cov, dcov = self.d01cov(p)
        
        return fisher(cov, dm, dcov)
    
    def natdlogp(self, p, x):
        """Natural gradient."""

        m = self.mean(p)
        dm = self.dmean(p)
        cov, dcov = self.d01cov(p)

        g = dlogp(x, m, cov, dm, dcov)
        fimat = fisher(cov, dm, dcov)

        return np.linalg.solve(fimat, g)