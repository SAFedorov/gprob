import itertools

import numpy as np
from numpy.linalg import LinAlgError

from .func import logp

# TODO: sampling would be very slow for large-aspect-ratio variables [[a0, a1, a2, a3, .. an]] for n >> 1
# TODO: remove normal-normal arithmetic operations
# TODO: clean up repr
# TODO: add power operator


class Elementary:

    id_counter = itertools.count()

    @staticmethod
    def create(n: int):
        return {next(Elementary.id_counter): i for i in range(n)}
    
    @staticmethod
    def union(iids1: dict, iids2: dict):
        """Ordered union of two dictionaries of iids."""

        diff = set(iids2) - set(iids1)   
        offs = len(iids1)

        union_iids = iids1.copy()
        union_iids.update({xi: (offs + i) for i, xi in enumerate(diff)}) 

        return union_iids
    
    @staticmethod
    def uunion(*args):
        """Unordered union of multiple dictionaries of elementary variables."""
        s = set().union(*args)
        return {k: i for i, k in enumerate(s)}
    
    @staticmethod
    def longer_first(op1, op2):
        (_, iids1), (_, iids2) = op1, op2

        if len(iids1) >= len(iids2):
            return op1, op2, False
        
        return op2, op1, True

    @staticmethod
    def complete_maps(op1, op2):

        (a1, iids1), (a2, iids2) = op1, op2

        if iids1 is iids2:
            return a1, a2, iids1

        (a1, iids1), (a2, iids2), swapped = Elementary.longer_first(op1, op2)
            
        union_iids = Elementary.union(iids1, iids2)
        a1_ = Elementary.pad_map(a1, len(union_iids))
        a2_ = Elementary.extend_map(a2, iids2, union_iids)

        if swapped:
            a1_, a2_ = a2_, a1_

        return a1_, a2_, union_iids
    
    @staticmethod
    def add(op1, op2):

        (a1, iids1), (a2, iids2) = op1, op2

        if iids1 is iids2:
            return a1 + a2, iids1

        (a1, iids1), (a2, iids2), _ = Elementary.longer_first(op1, op2)
            
        union_iids = Elementary.union(iids1, iids2)
        sum_a = Elementary.pad_map(a1, len(union_iids))

        idx = [union_iids[k] for k in iids2]
        sum_a[idx] += a2

        return sum_a, union_iids
    
    @staticmethod
    def pad_map(a, new_len):

        len_ = a.shape[0]
        new_shape = (new_len, *a.shape[1:])
        new_a = np.zeros(new_shape)
        new_a[:len_] = a

        return new_a

    @staticmethod
    def extend_map(a, iids: dict, new_iids: dict):

        new_shape = (len(new_iids), *a.shape[1:])
        new_a = np.zeros(new_shape)
        idx = [new_iids[k] for k in iids]
        new_a[idx] = a

        return new_a
    
    @staticmethod
    def join_maps(op1, op2):

        # Only works for 2D maps. Preserves the iids order.

        (a1, iids1), (a2, iids2) = op1, op2

        if iids1 is iids2:
            return a1 + a2, iids1

        (a1, iids1), (a2, iids2), swapped = Elementary.longer_first(op1, op2)
            
        union_iids = Elementary.union(iids1, iids2)
        l1, l2 = a1.shape[1], a2.shape[1]
        cat_a = np.zeros((len(union_iids), l1 + l2))

        idx = [union_iids[k] for k in iids2]

        if swapped:
            cat_a[:len(iids1), l2:] = a1
            cat_a[idx, :l2] = a2
        else:
            cat_a[:len(iids1), :l1] = a1
            cat_a[idx, l1:] = a2

        return cat_a, union_iids
    
    @staticmethod
    def ujoin_maps(ops):
        # ops is a sequence ((a1, iids1), (a2, iids2), ...), where `a`s are 
        # strictly two-dimensional matrices.  

        union_iids = Elementary.uunion(*[iids for _, iids in ops])

        dims = [a.shape[1] for a, _ in ops]
        cat_a = np.zeros((len(union_iids), sum(dims)))
        n1 = 0
        for i, (a, iids) in enumerate(ops):
            n2 = n1 + dims[i]
            idx = [union_iids[k] for k in iids]
            cat_a[idx, n1: n2] = a
            n1 = n2

        return cat_a, union_iids


class Normal:

    __slots__ = ("a", "b", "iids", "size", "shape", "ndim")
    __array_ufunc__ = None

    def __init__(self, a, b, iids=None):
        if a.shape[1:] != b.shape:
            a = np.broadcast_to(a, (a.shape[0], *b.shape))

        self.a = a
        self.b = b

        self.size = b.size
        self.shape = b.shape
        self.ndim = b.ndim

        if iids is None:
            iids = Elementary.create(a.shape[0])
        elif len(iids) != a.shape[0]:
            raise ValueError(f"The length of iids ({len(iids)}) does not match "
                             f"the outer dimension of `a` ({a.shape[0]}).")

        self.iids = iids  # Dictionary {id -> index, ...}

    @property
    def _a2d(self): return self.a.reshape((self.a.shape[0], self.b.size))
    
    @property
    def _b1d(self): return self.b.reshape((self.b.size,))

    def __repr__(self):
        if self.size == 0:
            return "" 
        
        if self.size == 1:
            return f"~ normal({self.mean():0.3g}, {self.var():0.3g})"
        
        return f"~ normal(..., shape={self.shape})"
    
    def __len__(self):
        return len(self.b)

    def __neg__(self):
        return Normal(-self.a, -self.b, self.iids)

    def __add__(self, other):
        if isinstance(other, Normal):
            a, iids = Elementary.add((self.a, self.iids), (other.a, other.iids))
            return Normal(a, self.b + other.b, iids)
        
        return Normal(self.a, self.b + other, self.iids)

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        if isinstance(other, Normal):
            a, iids = Elementary.add((self.a, self.iids), (-other.a, other.iids))
            return Normal(a, self.b + other.b, iids)
        
        return Normal(self.a, self.b - other, self.iids)
    
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, Normal):
            # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
            # for  x = <x> + dx  and  y = <y> + dy.

            a, iids = Elementary.add((self.a * other.b, self.iids),
                                     (other.a * self.b, other.iids))
            b = self.b * other.b

            return Normal(a, b, iids)
        
        return Normal(self.a * other, self.b * other, self.iids)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Normal):
            # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
            # for  x = <x> + dx  and  y = <y> + dy.

            a, iids = Elementary.add((self.a / other.b, self.iids),
                                     (other.a * (-self.b) / other.b**2, other.iids))
            b = self.b / other.b

            return Normal(a, b, iids)
        
        return Normal(self.a / other, self.b / other, self.iids)
    
    def __rtruediv__(self, other):
        # Linearized fraction  x/y = <x>/<y> - dy<x>/<y>^2,
        # for  x = <x>  and  y = <y> + dy.
        # Only need co cover the case when `other`` is a number or sequence.
        
        a = self.a * (-other) / self.b**2
        b = other / self.b
        return Normal(a, b, self.iids)
    
    # TODO:
    #def __matmul__(self, other):
    #    pass

    #def __rmatmul__(self, other):
    #    return Normal(other @ self.a, other @ self.b, self.iids)

    def __getitem__(self, key):
        a = self.a[..., key]
        b = self.b[key]
        return Normal(a, b, self.iids)

    def __setitem__(self, key, value):
        if self.iids == value.iids:
            self.a[..., key] = value.a
            self.b[key] = value.b
        else:
            raise ValueError("The iids of the assignment target and the operand"
                             " must be the same to assign at an index.")

    def __or__(self, observations: dict):
        """Conditioning operation.
        
        Args:
            observations: A dictionary of observations {variable: value, ...}, 
                where variables are normal random variables, and values can be 
                deterministic or random variables.
        
        Returns:
            Conditional normal variable.
        """

        cond = join([k-v for k, v in observations.items()])

        av, ac, union_iids = Elementary.complete_maps((self._a2d, self.iids), 
                                                      (cond._a2d, cond.iids))

        sol, _, rank, _ = np.linalg.lstsq(ac.T, -cond._b1d, rcond=None)  # TODO: right now SVD is performed twice on ac
        new_b = self._b1d + sol @ av

        if rank < cond.size:
            delta = (cond._b1d + ac.T @ sol) 
            res = delta @ delta
            eps = np.finfo(float).eps * max(ac.shape)  # TODO: need to multiply be the singular value

            if res > eps:
                raise RuntimeError("Conditions cannot be simultaneously satisfied.") 

        # Computes the projection of the column vectors of `a` on the subspace 
        # orthogonal to the constraints. 
        sol_, _, _, _ = np.linalg.lstsq(ac, av, rcond=None)
        new_a = av - ac @ sol_

        # Reshapes the result
        new_a = np.reshape(new_a, self.a.shape)
        new_b = np.reshape(new_b, self.b.shape)

        return Normal(new_a, new_b, union_iids)
    
    def __and__(self, other):
        """Combines two random variables into one vector."""

        if not isinstance(other, Normal):
            other = asnormal(other)

        if self.b.ndim > 1 or other.b.ndim > 1:
            raise ValueError("& operation is only applicable to 0- and 1-d arrays.")
        
        cat_a, iids = Elementary.join_maps((self._a2d, self.iids), 
                                           (other._a2d, other.iids))
        cat_b = np.concatenate([self._b1d, other._b1d], axis=0)

        return Normal(cat_a, cat_b, iids)  
    
    def __rand__(self, other):
        return self & other

    def mean(self):
        """Mean"""
        return self.b

    def var(self):
        """Variance"""
        var = np.einsum("ij, ij -> j", self._a2d, self._a2d).reshape()
        return var.reshape(self.shape)
    
    def cov(self):
        """Covariance"""
        a_ = self._a2d
        return a_.T @ a_
    
    def sample(self, n=None):
        """Samples the random variable `n` times."""
        # n=None returns scalar output

        if n is None:
            nshape = tuple()
        else:
            nshape = (n,)
        
        r = np.random.normal(size=(*nshape, len(self.iids)))
        return (r @ self._a2d + self._b1d).reshape((*nshape, *self.shape))  
    
    def logp(self, x):
        """Log likelihood of a sample.
    
        Args:
            x: Sample value or a sequence of sample values.

        Returns:
            Natural logarithm of the probability density at the sample value - 
            a single number for single sample inputs, and an array for sequence 
            inputs.
        """
        return logp(x, self._b1d, self.cov())


def join(args):
    """Combines several of random (and possibly deterministic) variables
    into one vector."""

    if isinstance(args, Normal):
        return args

    nvs = [asnormal(v) for v in args]
    ops = [(v._a2d, v.iids) for v in nvs]
    a, iids = Elementary.ujoin_maps(ops)
    b = np.concatenate([np.array(v.b, ndmin=1) for v in nvs])

    return Normal(a, b, iids)


def asnormal(v):
    if isinstance(v, Normal):
        return v

    # v is a number or sequence of numbers
    return Normal(a=np.array([]), b=np.array(v), iids={})


def normal(mu=0, sigmasq=1, size=1):
    """Creates a new normal random variable.
    
    Args:
        mu: scalar or vector mean value
        sigmasq: scalar variance or covariance matrix

    Returns:
        Normal random variable, scalar or vector.
    """

    sigmasq = np.array(sigmasq)
    mu = np.array(mu)

    if sigmasq.size == 1:

        sigmasq = sigmasq.reshape((1,))
        mu = mu.reshape(tuple())  # makes scalar
        
        if size == 1:
            # Single scalar variable
            if sigmasq < 0:
                raise ValueError("Negative scalar sigmasq")
            
            return Normal(np.sqrt(sigmasq), mu)
        
        # Vector of independent identically-distributed variables.
        return Normal(np.sqrt(sigmasq) * np.eye(size, size), mu * np.ones(size))

    try:
        atr = np.linalg.cholesky(sigmasq)
        return Normal(atr.T, mu)
    except LinAlgError:
        # The covariance matrix is not strictly positive-definite.
        pass

    # To handle the positive-semidefinite case, do the orthogonal decomposition. 
    eigvals, eigvects = np.linalg.eigh(sigmasq)  # sigmasq = V D V.T

    if (eigvals < 0).any():
        raise ValueError("Negative eigenvalue in sigmasq matrix.")
    
    atr = eigvects @ np.diag(np.sqrt(eigvals))

    return Normal(atr.T, mu)