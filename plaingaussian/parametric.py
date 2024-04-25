import numpy as np

import jax
from jax import numpy as jnp

from .normal import Normal
from .emaps import complete, ElementaryMap
from .func import logp, dlogp, fisher


def jmp(fun, primals, tangents):
    """Forward mode jacobain-matrix product for `fun`. Spans over the 0-th 
    dimension of each of the arrays in `tangents`, and stacks the results 
    the along the 0-th dimension of the output.
    
    Args:
        primals: A list or tuple of positional arguments to `fun` at which its 
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
    # `p` is limited to 1D with the scipy.minimize signature in mind

    if isinstance(input_vs, (list, tuple)):
        inbs = tuple(force_float(v.b) for v in input_vs)
        
        ems = complete([v.emap for v in input_vs])
        inas = tuple(em.a for em in ems)
        elem = ems[0].elem

        afun = lambda p: jmp(lambda *v: f(p, v), inbs, inas)
        bfun = lambda p: f(p, inbs)
    elif isinstance(input_vs, Normal):
        elem = input_vs.emap.elem
        afun = lambda p: jmp(lambda v: f(p, v), (force_float(input_vs.b),),
                             (input_vs.emap.a,))
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

    return ParametricNormal(afun, bfun, dafun, dbfun, elem)
    

class ParametricNormal:
    """Parametric normal random variable produced by a linearized function."""

    __slots__ = ("_afun", "_bfun", "_dafun", "_dbfun", "elem")

    def __init__(self, afun, bfun, dafun, dbfun, elem):
        # ... is the array shape of the variable mean, nelem is the number 
        # of the elementary variables, np is the number of the parameters.

        self._afun = afun  # output shape: nelem x ... 
        self._bfun = bfun  # output shape: ...
        self._dafun = dafun  # output shape: np x nelem x ... 
        self._dbfun = dbfun  # output shape: np x ...
        self.elem = elem

    def __call__(self, p):
        em = ElementaryMap(self.a(p), self.elem)
        return Normal(em, self.mean(p))

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
    

def force_float(x):
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64)
    return x