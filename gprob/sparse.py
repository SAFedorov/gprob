import numpy as np

import gprob.normal_ as gpn
from .normal_ import Normal, asnormal, print_normal, as_numeric_or_normal, broadcast_to
from .external import einsubs


def iid_repeat(x, nrep=1, axis=0):
    """Creates a sparse array of independent identically distributed copies of `x`."""

    # TODO: support multiple axes
    # TODO: broadcasting does not work when the axis != 0
    
    if axis < 0:
        axis = x.ndim + axis

    if isinstance(x, SparseNormal):
        iax = sorted([ax + 1 if ax >= axis else ax for ax in x.iaxes] + [axis])
        iax = tuple(iax)
        v = x.v
    else:
        iax = (axis,)
        v = asnormal(x)

    # Index that adds a new axis.
    idx = [slice(None, None, None)] * v.ndim
    idx.insert(axis, None)
    idx = tuple(idx)

    new_shape = v.shape[:axis] + (nrep,) + v.shape[axis:]
    v = broadcast_to(v[idx], new_shape)

    v = iid_copy(v)
    return SparseNormal(v, iax)


def iid_copy(x):
    """Creates an independent identically distributed duplicate of `x`."""
    # TODO: Make it a method function? Sparse Normals need to implement it on their own. 

    # Copies are needed for the case if the original array is modified 
    # in-place later. Such modifications should not affect the new variable.
    return Normal(x.a.copy(), x.b.copy())


def assparsenormal(x):
    if isinstance(x, SparseNormal):
        return x
    
    v = asnormal(x)
    return SparseNormal(v, tuple(range(v.ndim)))


def _lifted_binary_op(name):
    """Generates a binary operation that applies a function to the operands' v, 
    checks the conformity of their independence axes, 
    and produces a new sparse normal."""

    def f(self, other):
        other = assparsenormal(other)
        v = getattr(self.v, name)(other.v)  # TODO: implement via operators for automatic handling of NotImplemented
        
        if v is NotImplemented:  # TODO: does this make any sense?
            return NotImplemented
        
        _validate_iaxes([self, other])
        return SparseNormal(v, self.iaxes)
    return f


class SparseNormal:
    """Array of block-independent normal random variables."""

    __slots__ = ("v", "iaxes")
    __array_ufunc__ = None
    _normal_priority_ = 10
    
    def __init__(self, v, iaxes=tuple()):
        self.v = v
        self.iaxes = iaxes  # Ordered sequence of axes along which the array 
                            # elements are independent from each other.

    @property
    def size(self):
        return self.v.size
    
    @property
    def shape(self):
        return self.v.shape
    
    @property
    def ndim(self):
        return self.v.ndim
    
    @property
    def real(self):
        return SparseNormal(self.v.real, self.iaxes)
    
    @property
    def imag(self):
        return SparseNormal(self.v.imag, self.iaxes)
    
    @property
    def T(self):
        # TODO: when transpose() method is implemented, switch to calling it here
        iaxes = tuple(sorted([self.ndim - ax - 1 for ax in self.iaxes]))
        return SparseNormal(self.v.T, iaxes)
    
    @property
    def iscomplex(self):
        return self.v.iscomplex
    
    def __repr__(self):
        return print_normal(self, extra_attrs=("iaxes",))
    
    def __len__(self):
        return len(self.v)
    
    def __neg__(self):
        return SparseNormal(-self.v, self.iaxes)

    # TODO: add a test that all the binary operations defined 

    __add__ = _lifted_binary_op("__add__")
    __radd__ = _lifted_binary_op("__radd__")
    __sub__ = _lifted_binary_op("__sub__")
    __rsub__ = _lifted_binary_op("__rsub__")
    __mul__ = _lifted_binary_op("__mul__")
    __rmul__ = _lifted_binary_op("__rmul__")
    __truediv__ = _lifted_binary_op("__truediv__")
    __rtruediv__ = _lifted_binary_op("__rtruediv__")
    __pow__ = _lifted_binary_op("__pow__")
    __rpow__ = _lifted_binary_op("__rpow__")

    def __matmul__(self, other):
        other, isnormal = as_numeric_or_normal(other)

        # TODO: add priority check
        # self._validate_iaxes(other)

        if self.ndim - 1 in self.iaxes:
            raise NotImplementedError("Matrix multiplication along "
                                      "independence axes is not implemented.")
            
        return SparseNormal(self.v @ other, self.iaxes)

    def __rmatmul__(self, other):
        other, isnormal = as_numeric_or_normal(other)

        # TODO: add priority check
        ax = 0 if self.ndim == 1 else self.ndim-2
        if ax in self.iaxes:
            raise NotImplementedError("Matrix multiplication along "
                                      "independence axes is not implemented.")

        return SparseNormal(other @ self.v, self.iaxes)
    
    def __getitem__(self, key):
        out_ax = _parse_index_key(self, key)
        out_iax = (out_ax[in_ax] for in_ax in self.iaxes)
        iaxes = tuple(ax for ax in out_iax if ax is not None)
        return SparseNormal(self.v[key], iaxes)
        
    # ---------- array methods ----------

    # TODO

    def moveaxis(self, source, destination):
        pass

    def ravel(self, order="C"):
        if self.ndim > 1:
            raise NotImplementedError # TODO: give a more meaningful error message
        
        return SparseNormal(self.v.ravel(order=order), self.iaxes)
    
    def reshape(self, newshape, order="C"):
        
        b = self.b.reshape(newshape, order=order)
        em = self.emap.reshape(newshape, order=order)

        # reshing is used by stacking and concatenation functions. TODO 
        raise NotImplementedError
    
    @staticmethod
    def concatenate(arrays, axis):
        arrays = [assparsenormal(ar) for ar in arrays]
        iaxes = _validate_iaxes(arrays)
        v = Normal.concatenate([x.v for x in arrays], axis=axis)
        return SparseNormal(v, iaxes)
    
    @staticmethod
    def stack(arrays, axis):
        arrays = [assparsenormal(ar) for ar in arrays]
        iaxes = _validate_iaxes(arrays)
        iaxes = tuple(ax + 1 if ax >= axis else ax for ax in iaxes)
        v = Normal.stack([x.v for x in arrays], axis=axis)
        return SparseNormal(v, iaxes)
    
    @staticmethod
    def einsum(subs, op1, op2):
        def out_iaxes(op, insubs, outsubs):
            """Calculates the indices of the independence axes for
            the output operand."""
            
            for i, c in enumerate(insubs):
                if i in op.iaxes and c not in outsubs:
                    raise ValueError("Contraction over an independence"
                                     f" axis ({i}).")
            
            iaxes = tuple(outsubs.index(c) for i, c in enumerate(insubs) 
                          if i in op.iaxes)
            return iaxes
        
        # TODO: convert to sparse normal or numerical operands

        # Converts the subscripts to an explicit form.
        (insu1, insu2), outsu = einsubs.parse(subs, (op1.shape, op2.shape))
        subs = f"{insu1},{insu2}->{outsu}"

        if isinstance(op1, SparseNormal) and isinstance(op2, SparseNormal):
            return (SparseNormal.einsum(subs, op1, op1.v.b)
                    + SparseNormal.einsum(subs, op1.v.b, op1))

        if isinstance(op1, SparseNormal) and not isinstance(op2, SparseNormal):
            iaxes = out_iaxes(op1, insu1, outsu)
            v = Normal.einsum(subs, op1.v, op2)
            return SparseNormal(v, iaxes)

        if isinstance(op2, SparseNormal) and not isinstance(op1, SparseNormal):
            iaxes = out_iaxes(op2, insu2, outsu)
            v = Normal.einsum(subs, op1, op2.v)
            return SparseNormal(v, iaxes)
        
        # The default that is not supposed to be reached, usually.
        return Normal.einsum(subs, op1, op2)

    # ---------- probability-related methods ----------

    # TODO: add a test that the doc strings are in sync with those of Normal --------------------

    def condition(self, observations, mask=None):  # TODO-----------------------------
        raise NotImplementedError
    
    def mean(self):
        """Mean"""
        return self.v.b

    def var(self):
        """Variance, `<(x-<x>)(x-<x>)^*>` where `*` is complex conjugation."""
        return self.v.var()

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
        raise NotImplementedError  # TODO-----------------------------------------------
    
    def sample(self, n=None):
        """Samples the random variable `n` times."""
        # n=None returns scalar output

        raise NotImplementedError  # TODO-----------------------------------------------
        

    def logp(self, x):
        """Log likelihood of a sample.
    
        Args:
            x: Sample value or a sequence of sample values.

        Returns:
            Natural logarithm of the probability density at the sample value - 
            a single number for single sample inputs, and an array for sequence 
            inputs.
        """
        raise NotImplementedError  # TODO-----------------------------------------------


def _validate_iaxes(seq):  # TODO: rename, maybe, since now it returns a value-------
    iaxs = set([x.iaxes for x in seq if len(x.v.emap.elem) !=0])
    
    if len(iaxs) == 1:
        return iaxs.pop()
    elif len(iaxs) == 0:
        return tuple()
    
    # len > 1
    raise ValueError("Only sparse normals with identical independence "
                     "axes can be combined in operations.")


def _parse_index_key(x, key):
    """Calculates the numbers of the output axes for each input axis based 
    on the key and validates the key."""

    if not isinstance(key, tuple):
        key = (key,)

    has_ellipsis = False
    has_bool = False
    
    out_ax = []
    new_dim = 0

    for k in key:
        if isinstance(k, int):
            out_ax.append(None)
            continue

        if isinstance(k, slice):
            out_ax.append(new_dim)
            new_dim += 1
            continue
        
        if k is Ellipsis:
            if has_ellipsis:
                raise IndexError("An index can only have a single "
                                 "ellipsis ('...').")
            
            has_ellipsis = True
            ellipsis_pos = len(out_ax)
            ellipsis_dim = new_dim
            continue

        if k is np.newaxis:
            new_dim += 1
            continue

        # Advanced indices. The limitations compared to numpy are that 
        # integer arrays are not allowed, and it is not allowed to combine 
        # multiple fancy indices in one key.
         
        ak = np.asarray(k)
        if np.issubdtype(ak.dtype, bool):
            if has_bool:
                raise IndexError("Only one bool index is allowed in a key.")
            else:
                has_bool = True

            used_dims = set(range(len(out_ax), len(out_ax) + ak.ndim))
            iaxes = set(x.iaxes)
            
            if not (used_dims.issubset(iaxes) or used_dims.isdisjoint(iaxes)):
                raise IndexError(f"Mixing independece {x.iaxes} and regular "
                                 "axes by the bool index affecting "
                                 f"the dimensions {tuple(sorted(used_dims))}.")

            out_ax.extend(new_dim for _ in range(ak.ndim))
            new_dim += 1
            continue

        raise IndexError("Only integers, slices (':'), ellipsis ('...'), "
                         "numpy.newaxis ('None') and boolean arrays are "
                         "valid indices.")
    
    # Checks if any input dimensions remain unconsumed.
    delta = x.ndim - len(out_ax)
    if delta > 0:
        if has_ellipsis:
            for i in range(ellipsis_pos, len(out_ax)):
                if out_ax[i] is not None:
                    out_ax[i] += delta

            out_ax[ellipsis_pos: ellipsis_pos] = range(ellipsis_dim, 
                                                       ellipsis_dim + delta)
        else:
            out_ax.extend(range(new_dim, new_dim + delta))
    elif delta < 0:
        raise IndexError(f"Too many indices: the array is {x.ndim}-dimensional,"
                         f" but {len(out_ax)} indices were given.")

    return out_ax