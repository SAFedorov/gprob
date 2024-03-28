import functools

import numpy as np

import jax
from jax import numpy as jnp
#jax.config.update("jax_enable_x64", True)  TODO: delete after veryfying that it is not needed here

from normal import Normal
from func import logp, dlogp, fisher


def jmp(f, primals, tangents):
    """Jacobain-matrix product."""

    # Note: the input and the output matrix are transposed, because vmap works
    # the fastest along the inner-most axis.
    #
    # The function can handle multiple arguments, but otherwise the same as in
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

    jvp = lambda t: jax.jvp(f, primals, t)[1]
    return jax.vmap(jvp)(tangents)


def pgmap(f):
    """Creates a parametric Gaussain map from a function with signature `f(p, rv)`."""
    return ParametricGaussianMap(f)


class ParametricGaussianMap:
    """Parametric Gaussain map."""

    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f

    def __call__(self, p, rv):
        # TODO: support calls with rv as normal arrays - just call f

        pn = self.partial(rv)
        return pn(p)

    def partial(self, rv):
        # TODO: add option to specify p instead of rv here (but only one argument at a time).
        # TODO: handle the case of a single random variable

        return ParametricGaussianMap._make_pn(self.f, *rv)  # The purpose of this is caching
        
        
    # TODO: add validation of caching behavior, in terms of the new variable generation, and leakage of the generated cov_jvps
    
    @staticmethod
    @functools.lru_cache
    def _make_pn(f, *input_rvs): # Unwrap to cach based on rv's, not their container
        
        # TODO: temporary solution, need to introduce a split function
        inbs, inas, iids = _get_ab(*input_rvs)

        atfun = lambda p: jmp(lambda *v: f(p, v), inbs, inas)  # gives a.T 
        datfun = lambda p: jmp(atfun, (jnp.array(p),), (jnp.eye(len(p)),))  #datfun = jax.jacfwd(atfun)
        bfun = lambda p: f(p, inbs)
        dbfun = lambda p: jmp(bfun, (jnp.array(p),), (jnp.eye(len(p)),)) #jax.jacfwd(bfun)

        # Note: the conversion of `p` to jnp array is necessary for the 
        # derivatives over `p` because the value of `p` input by the user can be 
        # not only an array, but also a list or tuple, while the behavior 
        # of jax.jvp is sensitive to the data type. In contrast, in `atfun` and 
        # `bfun` the inputs to jax.jvp always are already always arrays.

        atfun = jax.jit(atfun)
        datfun = jax.jit(datfun)
        bfun = jax.jit(bfun)
        dbfun = jax.jit(dbfun)

        return ParametricNormal(atfun, bfun, datfun, dbfun, iids)


def _get_ab(*input_rvs):
    rv_means = tuple(v.b for v in input_rvs)

    s = set().union(*[v.iids.keys() for v in input_rvs])
    iids = {k: i for i, k in enumerate(s)}  # The values of iids must be range(len(iids)) 
    rv_as = tuple(v._extended_map(iids).T for v in input_rvs)  # nrv x rv_dim x niids  # TODO: move this inside "compatible maps"?

    return rv_means, rv_as, iids
    

class ParametricNormal:
    """Parametric normal random variable produced by a Gaussian map."""

    # TODO: annotate the matrix shapes
    # define _cov_and_dcov (or call it _d01cov)

    __slots__ = ("atfun", "bfun", "datfun", "dbfun", "iids")

    def __init__(self, atfun, bfun, datfun, dbfun, iids):
        self.atfun = atfun
        self.bfun = bfun
        self.datfun = datfun
        self.dbfun = dbfun
        self.iids = iids

    def __call__(self, p):
        return Normal(self.a(p), self.mean(p), self.iids)

    # TODO: Note converts to normal array

    def a(self, p):
        at = np.array(self.atfun(p))  # nf x nrv
        return at.T 

    def da(self, p):
        dat = np.array(self.datfun(p))  
        return dat.transpose(0, 2, 1)  # np x nf x niids
    
    def mean(self, p):
        return np.array(self.bfun(p))
    
    def dmean(self, p):
        return np.array(self.dbfun(p))  # np x nf
    
    def cov(self, p):
        a = self.a(p)
        return a @ a.T

    def d01cov(self, p):
        """Covariance matrix with its derivative.

        Args:
            p: An array or sequence of parameter values.
        
        Returns:
            (covariance, dcovariance/dp)    
        """
        
        a = self.a(p)
        cov = a @ a.T
        da = self.da(p)
        prod1 = da @ a.T
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