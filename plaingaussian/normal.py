import numpy as np
import scipy as sp
from scipy.linalg import lapack

import itertools

# TODO: sampling and logp would be very slow for large-aspect-ratio variables [[a0, a1, a2, a3, .. an]] for n >> 1
# TODO: remove normal-normal arithmetic operations
# TODO: move the meat of logp into the functional file

# TODO: make it a named tuple?

class Normal:
    """Vector-valued normally-distributed random variable,
    
    v = a @ iids + b,
    
    where iids are independent identically-distributed Gaussian variables, 
    iids[k] ~ N(0, 1) for all k.
    """

    __slots__ = ("a", "b", "iids", "dim")
    __array_ufunc__ = None
    id_counter = itertools.count()

    def __init__(self, a, b, iids=None):
        if a.shape[-2] != b.shape[-1]:
            raise ValueError("The shapes of a and b do not agree. The shape of "
                             f"a is {a.shape} and the shape of b is {b.shape}.")

        self.a = a  # matrix defining the linear map iids -> v
        self.b = b  # mean vector
        self.dim = a.shape[-2]

        if iids is None:
            # Allocates new independent random variables.
            iids = {next(Normal.id_counter): i for i in range(a.shape[-1])}
        elif len(iids) != a.shape[-1]:
            raise ValueError(f"The length of iids ({len(iids)}) does not match "
                             f"the inner dimension of a ({a.shape[-1]}).")

        self.iids = iids  # Dictionary {id -> column_index, ...}
    
    def _extended_map(self, new_iids: dict):
        """Extends `self.a` to a new set of iid variables that must be 
        a superset of its current iids.
        
        Args:
            new_iids: A dictionary {id -> column_index, ...}, which satisfies 
                `list(new_iids.values()) == list(range(len(new_iids)))`
        
        Returns:
            A new `a` matrix.
        """
        
        new_a = np.zeros((self.dim, len(new_iids)))
        idx = [new_iids[k] for k in self.iids]

        new_a[:, idx] = self.a
        # For python >= 3.6, the dictionaries are order-preserving, 
        # which means that the values for iids are always their sequential 
        # numers. If the dictionaries are not order-preserving, can use this:
        # new_a[:, idx] = np.take(self.a, list(self.iids.values()), axis=1)
        # which is only 20% slower in my benchmark.
        
        return new_a
    
    def _compatible_maps(self, other):
        """Extends `self.a` and `other.a` to the union of their iids."""

        if self.iids is other.iids:
            # The maps operating on the same iids are already compatible.
            return self.a, other.a, self.iids

        # The largest go first, because its iid variable map remains unmodified.
        if len(self.iids) >= len(other.iids):
            op1, op2 = self, other
        else:
            op1, op2 = other, self
            
        s2m1 = set(op2.iids) - set(op1.iids)   
        offs = len(op1.iids)
        new_iids = op1.iids.copy()
        new_iids.update({xi: (offs + i) for i, xi in enumerate(s2m1)}) 

        a1 = np.pad(op1.a, ((0, 0), (0, len(s2m1))), 
                    "constant", constant_values=(0,))
        a2 = op2._extended_map(new_iids)

        if op1 is other:
            a1, a2 = a2, a1  # Swap the order back to (self, other)

        return a1, a2, new_iids
    
    def __mul__(self, other):
        if isinstance(other, Normal):
            # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
            # for  x = <x> + dx  and  y = <y> + dy.
            
            a1, a2, new_iids = self._compatible_maps(other)
            a = (a1.T * other.b + a2.T * self.b).T
            b = self.b * other.b

            return Normal(a, b, new_iids)
        
        # Scalar variables span over sequences
        if self.dim == 1 and isinstance(other, np.ndarray) and other.ndim == 1:
            return Normal(np.outer(other, self.a), self.b * other, self.iids)
        
        if self.dim == 1 and isinstance(other, (list, tuple)):
            return Normal(np.outer(other, self.a), self.b * other, self.iids)

        # Otherwise other must be a number or vector of numbers
        return Normal(self.a * other, self.b * other, self.iids)
    
    def __truediv__(self, other):
        if isinstance(other, Normal):
            # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
            # for  x = <x> + dx  and  y = <y> + dy.
            
            a1, a2, new_iids = self._compatible_maps(other)
            a = (a1.T / other.b - a2.T * self.b / other.b**2).T
            b = self.b / other.b

            return Normal(a, b, new_iids)

        # TODO: add sequences
        
        return Normal(self.a / other, self.b / other, self.iids)
    
    def __rtruediv__(self, other):
        # Linearized fraction  x/y = <x>/<y> - dy<x>/<y>^2,
        # for  x = <x>  and  y = <y> + dy.
        
        a = - (self.a.T * other / self.b**2).T
        b = other / self.b
        return Normal(a, b, self.iids)
    
    def __add__(self, other):
        if not isinstance(other, Normal):
            # Other must be a number or numeric vector.
            return Normal(self.a, self.b + other, self.iids)

        a1, a2, new_iids = self._compatible_maps(other)
        return Normal(a1 + a2, self.b + other.b, new_iids)

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        if not isinstance(other, Normal):
            # Assuming that other is a number or numeric vector.
            return Normal(self.a, self.b - other, self.iids)

        a1, a2, new_iids = self._compatible_maps(other)
        return Normal(a1 - a2, self.b - other.b, new_iids)
    
    def __rsub__(self, other):
        return (-1) * self + other
    
    def __neg__(self):
        return (-1) * self
    
    def __repr__(self):

        #TODO: add tests, for zero dimensions and empty matrices

        if self.dim == 0:
            return ""
        
        if self.dim == 1:
            mu = self.b[0]
            sigmasq = (self.a @ self.a.T)[0, 0] if self.a.size != 0 else 0
            return f"~ normal({mu:0.3g}, {sigmasq:0.3g})"
        
        # Better-looking for larger dimensions
        return f"~ normal\na:\n{self.a}\nb:\n{self.b}"

    def __getitem__(self, key):
        # TODO: add checks for the return formats of different keys
        a = np.array(self.a[key], ndmin=2)
        b = np.array(self.b[key], ndmin=1)

        return Normal(a, b, self.iids)

    def __setitem__(self, key, value):
        if self.iids == value.iids:
            self.a[key] = value.a
            self.b[key] = value.b
        else:
            raise ValueError("The iids of the assignment target and the operand"
                             " must be the same to assign at an index.")   
    
    def __len__(self): 
        # Defining length makes the class interact slower with numpy arrays 
        # (inclusing scalars) upon right-multiplication, because the class 
        # starts looking like a sequence type.
        raise NotImplementedError() # TODO should rather set __len__ = None
    
    def __or__(self, observations: dict):
        """Conditioning operation.
        
        Args:
            observations: A dictionary of observations {variable: value, ...}, 
                where variables are normal random variables, and values can be 
                deterministic or random variables.
        
        Returns:
            Conditional normal variable.
        """

        condition = join([k-v for k, v in observations.items()])
        av, ac, new_iids = self._compatible_maps(condition)

        sol_b, res, _, _ = np.linalg.lstsq(ac, -condition.b, rcond=None) 
        new_b = self.b + np.dot(av, sol_b)

        if res.size != 0:
            raise RuntimeError("Conditions cannot be simultaneously satisfied.")  # TODO: do not raise this error if the residual is zero 

        # Computes the projection of the a vectors on the subspace orthogonal 
        # to the constraints. 
        sol_a, _, _, _ = np.linalg.lstsq(ac.T, av.T, rcond=None)
        new_a = av - np.dot(sol_a.T, ac)

        return Normal(new_a, new_b, new_iids)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""

        a1, a2, new_iids = self._compatible_maps(other)
        new_a = np.concatenate([a1, a2], axis=0)
        new_b = np.concatenate([self.b, other.b], axis=0)
        return Normal(new_a, new_b, new_iids)
    
    def __rmatmul__(self, other):
        return Normal(other @ self.a, other @ self.b, self.iids)

    def mean(self):
        """Mean"""
        if self.dim == 1:
            return self.b[0]
        return self.b

    def var(self):
        """Variance"""
        variance = np.einsum("ij, ij -> i", self.a, self.a)
        if self.dim == 1:
            return variance[0]
        return variance
    
    def cov(self):
        """Covariance"""
        return self.a @ self.a.T
    
    def sample(self, n=1):
        r = np.random.normal(size=(len(self.iids), n))
        samples = np.dot(self.a, r).T + self.b

        # The return formats differ depending on the dimension.
        if self.dim == 1:
            if n == 1:
                return samples[0, 0]

            return samples[:, 0]

        if n == 1:
            return samples[0, :]
        
        return samples 

    def logp(self, x):
        """Log likelihood of a sample.
        
        Args:
            x: Sample value.

        Returns:
            Natural logarithm of the probability density at the sample value.
        """

        # TODO: return batching?

        x = np.array(x, ndmin=1)

        if len(x) != self.dim or x.ndim != 1:
            # Raises error because relying on Numpy broadcusting rules in this 
            # case can result in counter-intuitive behavior like constants  
            # being treated as vectors with all identical components.

            raise ValueError("The dimension of the sample vector "
                             f"({len(x)}) does not match the dimension " 
                             f"of the random variable ({self.dim}).") 
        
        covm = self.a @ self.a.T
        lt, status = lapack.dpotrf(covm, lower=True)

        if status == 0:
            sol = sp.linalg.solve_triangular(lt, x - self.b, 
                                             check_finite=False, lower=True)
            
            rank = self.dim  # If the solution suceeded, the rank is full.
            log_sqrt_det = np.sum(np.log(np.diag(lt)))
        
        else:
            # The covariance matrix is probably degenerate.
             
            sol, res, rank, sv = np.linalg.lstsq(self.a, x - self.b, rcond=None)
            sv = sv[:rank]  # Selects only non-zero singular values.
            log_sqrt_det = np.sum(np.log(sv))

        norm = np.log(np.sqrt(2 * np.pi)) * rank + log_sqrt_det
        llk = - sol @ sol / 2 - norm  # log likelihood

        if rank == self.dim:
            # The solution must be good.
            return llk
        
        # If the rank is not full, checks the residual error.
        # May need to calculate it, as it is not always returned by lstsq.
        
        if res.size == 0:
            delta = sol - (x - self.b)
            res = (delta @ delta,)  # Put in tuple for shape compatibility.

        eps = np.finfo(float).eps * max(self.a.shape[-2:]) 

        if np.abs(res[0]) > eps:
            # The input is a zero-probability sample.
            llk = float("-inf")

        return llk


def join(*args):
    """Combines several of random (and possibly deterministic) variables
    into one vector."""

    if len(args) == 0:
        raise ValueError("Zero arguments cannot be joined.")

    if len(args) == 1:
        if isinstance(args[0], (tuple, list)):
            vs = args[0]
        else:
            return as_normal(args[0])
    else:
        vs = args

    vsl = [as_normal(v) for v in vs]

    s = set().union(*[v.iids.keys() for v in vsl])
    iids = {k: i for i, k in enumerate(s)}  # The values of iids must be range(len(iids)) TODO: move this to Normal class 

    a = np.concatenate([v._extended_map(iids) for v in vsl], axis=0)  # TODO: Is there a better way of doing it?
    b = np.concatenate([v.b for v in vsl])

    return Normal(a, b, iids)


def as_normal(v):
    if isinstance(v, Normal):
        return v

    # v is a number or sequence of numbers
    return Normal(a=np.array([[]]), b=np.array(v, ndmin=1))


# TODO: rename to normal()

def N(mu=0, sigmasq=1, dim=1, lu=None):
    """Creates a new normal random variable.
    
    Args:
        mu: scalar or vector mean value
        sigmasq: scalar variance or covariance matrix

    Returns:
        Normal random variable, scalar or vector.
    """

    sigmasq = np.array(sigmasq, ndmin=2)
    mu = np.array(mu, ndmin=1)

    # Handles the scalar case when mu and sigmasq are simple numbers 
    if sigmasq.shape == (1, 1):
        if dim == 1:
            # Single scalar variable
            if sigmasq[0, 0] < 0:
                raise ValueError("Negative scalar sigmasq")
            return Normal(np.sqrt(sigmasq), mu)
        
        # Vector of independt identically-distributed variables
        return Normal(np.sqrt(sigmasq) * np.eye(dim, dim), mu * np.ones(dim))
    
    # If sigmasq is not a scalar, the external value of the argument is ignored.
    dim = sigmasq.shape[0]

    if len(mu) != dim:
        mu = mu * np.ones(dim)  # Expands the dimensions of mu. 
                                 # This allows, in particular, to not explicitly 
                                 # supply mu when creating zero-mean vector 
                                 # variables.      

    if (lu is None) or lu:
        try:
            # LU decomposition
            a = np.linalg.cholesky(sigmasq)
            return Normal(a, mu)
        except np.linalg.LinAlgError as e:  # TODO: just LinAlgError
            # LU decomposition fails if the covariance matrix is not strictly
            # positive-definite, while we also allow positive-semidefinite
            # matrices, unless lu=True.
            if lu is True:
                raise e

    # TODO: change to SVD? Then the matrix structure will always be the same, and lu argument won't be needed. Need to time just for curiosity.

    # If lu is False, or the LU decomposition failed, 
    # do the orthogonal decomposition sigmasq = Q D Q'
    eigvals, eigvects = np.linalg.eigh(sigmasq)

    if (eigvals < 0).any():
        raise ValueError("Negative eigenvalue in sigmasq matrix")
    
    a = eigvects @ np.diag(np.sqrt(eigvals))

    return Normal(a, mu)