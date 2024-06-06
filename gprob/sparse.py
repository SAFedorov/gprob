import numpy as np

from .normal_ import (Normal, asnormal, print_normal, as_numeric_or_normal, 
                      broadcast_to)
from .external import einsubs


# TODO: as_numeric_or_normal, can it be simplified away? -------------------------------

def iid_repeat(x, nrep=1, axis=0):
    """Creates a sparse array of independent identically distributed 
    copies of `x`."""
    
    if axis < 0:
        axis = x.ndim + axis + 1

    x = assparsenormal(x)
    x = x.iid_copy()

    iax = sorted([ax + 1 if ax >= axis else ax for ax in x.iaxes] + [axis])
    iax = tuple(iax)
    v = x.v

    # Index that adds a new axis.
    idx = [slice(None, None, None)] * v.ndim
    idx.insert(axis, None)
    idx = tuple(idx)

    new_shape = v.shape[:axis] + (nrep,) + v.shape[axis:]
    v = broadcast_to(v[idx], new_shape)

    return SparseNormal(v, iax)


def assparsenormal(x):
    if isinstance(x, SparseNormal):
        return x
    
    if (hasattr(x, "_normal_priority_") 
        and x._normal_priority_ > SparseNormal._normal_priority_):

        raise TypeError(f"The variable {x} cannot be converted to "
                        "a sparse normal variable because it is already "
                        f"of higher priority ({x._normal_priority_} "
                        f"> {SparseNormal._normal_priority_}).")
    
    v = asnormal(x)
    return SparseNormal(v, tuple())


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
        return self.transpose()
    
    @property
    def iscomplex(self):
        return self.v.iscomplex
    
    def __repr__(self):
        return print_normal(self, extra_attrs=("iaxes",))
    
    def __len__(self):
        return len(self.v)
    
    def __neg__(self):
        return SparseNormal(-self.v, self.iaxes)

    def __add__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v + other.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __radd__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v + self.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __sub__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v - other.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __rsub__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v - self.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __mul__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v * other.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __rmul__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v * self.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __truediv__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v / other.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __rtruediv__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v / self.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __pow__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v ** other.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

    def __rpow__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v ** self.v
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)

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

        # TODO: add priority check ----------------------------------------------------------------------------
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

    def conjugate(self):
        return SparseNormal(self.v.conjugate(), self.iaxes)
    
    def conj(self):
        return self.conjugate()
    
    def cumsum(self, axis):
        if not np.issubdtype(np.array(axis).dtype, int):
            # For regular arrays, None is the default value. For sparse arrays, 
            # it is not supported because these arrays usually 
            # cannot be flattened.

            raise ValueError("Axis must be an integer.")
        
        if axis < 0:
            axis = self.ndim + axis
        
        if axis in self.iaxes:
            raise ValueError("The computation of cumulative sums along "
                             "independence axes is not supported.")

        return SparseNormal(self.v.cumsum(axis), self.iaxes)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        if axis1 < 0:
            axis1 = self.ndim + axis1

        if axis2 < 0:
            axis2 = self.ndim + axis2

        if (axis1 in self.iaxes) or (axis2 in self.iaxes):
            raise ValueError("Taking diagonals along independence axes "
                             "is not supported.")
        iaxes = []
        for ax in self.iaxes:
            ax_ = ax
            if ax > axis1:
                ax_ -= 1
            if ax > axis2:
                ax_ -= 1
            iaxes.append(ax_)

        iaxes = tuple(iaxes)
        return SparseNormal(self.v.diagonal(offset, axis1, axis2), iaxes)

    def flatten(self, order="C"):
        if not self.iaxes or self.ndim <= 1:
            return SparseNormal(self.v.flatten(order=order), self.iaxes)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            if max_dim in self.iaxes:
                iaxes = (0,)
            else:
                iaxes = tuple()

            return SparseNormal(self.v.flatten(order=order), iaxes)
        
        raise ValueError("Sparse normal variables cannot be flattened unless "
                         "they have no independence axes, "
                         "their number of dimensions is <=1 or "
                         "all dimensions except maybe one have size <=1.")
    
    def moveaxis(self, source, destination):
        if source < 0:
            source = self.ndim + source

        if destination < 0:
            destination = self.ndim + destination
        
        iaxes = []
        for ax in self.iaxes:
            if ax == source:
                ax = destination
            elif ax > source and ax <= destination:
                ax -= 1
            elif ax >= destination and ax < source:
                ax += 1
            
            iaxes.append(ax)
        iaxes = tuple(sorted(iaxes))

        return SparseNormal(self.v.moveaxis(source, destination), iaxes)
    
    def ravel(self, order="C"):
        if not self.iaxes or self.ndim <= 1:
            return SparseNormal(self.v.ravel(order=order), self.iaxes)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            if max_dim in self.iaxes:
                iaxes = (0,)
            else:
                iaxes = tuple()

            return SparseNormal(self.v.ravel(order=order), iaxes)
        
        raise ValueError("Sparse normal variables cannot be flattened unless "
                         "they have no independence axes, "
                         "their number of dimensions is <=1 or "
                         "all dimensions except maybe one have size <=1.")
    
    def reshape(self, newshape, order="C"):

        v = self.v.reshape(newshape, order)  
        # Reshaping the underlying varibale before the axes check is
        # to yield a meaningful error message if the shapes are inconsistent.

        if v.size == 0:
            # The transformation of independence axes for zero-size variables 
            # cannot be determined unambiguously, so we always assume that 
            # the transformed variable has no independence axes.
            return SparseNormal(v, tuple())

        new_dim = 0

        new_cnt = 1
        old_cnt = 1

        iaxes = []
        for i, n in enumerate(self.shape):
            if i in self.iaxes:
                if n != 1:
                    # Skips all trivial dimensions first.
                    while new_dim < len(newshape) and newshape[new_dim] == 1:
                        new_dim += 1

                if (new_dim < len(newshape) and newshape[new_dim] == n 
                    and new_cnt == old_cnt):
                    
                    iaxes.append(new_dim)
                else:
                    raise ValueError("Reshaping that affects independence axes "
                                     f"is not supported. Axis {i} is affected "
                                     "by the requested shape transformation "
                                     f"{self.shape} -> {newshape}.")
                
                old_cnt *= n
                new_cnt *= newshape[new_dim]
                new_dim += 1
            else:
                old_cnt *= n

                while new_cnt < old_cnt:
                    new_cnt *= newshape[new_dim]
                    new_dim += 1
        
        iaxes = tuple(sorted(iaxes))
        return SparseNormal(v, iaxes)
    
    def split(self, indices_or_sections, axis=0):
        if axis < 0:
            axis = self.ndim + axis

        if axis in self.iaxes:
            raise ValueError("Splitting along independence axes "
                             "is not supported.")
        
        vpieces = self.v.split(indices_or_sections, axis)
        return [SparseNormal(v, self.iaxes) for v in vpieces]
    
    def sum(self, axis, keepdims=False):
        if not isinstance(axis, int) and not isinstance(axis, tuple):
            raise ValueError("`axis` must be an integer or "
                             "a tuple of integers.")
            # None, the default value for non-sparse arrays, is not supported,
            # because in the typical case the variable has at least one 
            # independence axis that cannot be contracted.

        if not isinstance(axis, tuple):
            axis = (axis,)

        sum_axes = [self.ndim + ax if ax < 0 else ax for ax in axis]

        if any(ax in self.iaxes for ax in sum_axes):
            raise ValueError("The computation of sums along "
                             "independence axes is not supported.")
        
        if keepdims:
            iaxes = self.iaxes
        else:
            iaxes = tuple(iax - [sax < iax for sax in sum_axes].count(True) 
                          for iax in self.iaxes) 

        v = self.v.sum(axis=axis, keepdims=keepdims)
        return SparseNormal(v, iaxes)

    def transpose(self, axes=None):
        if axes is None:
            iaxes = [self.ndim - ax - 1 for ax in self.iaxes]
        else:
            axes_ = [self.ndim + ax if ax < 0 else ax for ax in axes]
            iaxes = [axes_.index(ax) for ax in self.iaxes]

        iaxes = tuple(sorted(iaxes))
        return SparseNormal(self.v.transpose(axes), iaxes)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        if axis1 < 0:
            axis1 = self.ndim + axis1

        if axis2 < 0:
            axis2 = self.ndim + axis2

        if (axis1 in self.iaxes) or (axis2 in self.iaxes):
            raise ValueError("Traces along independence axes "
                             "are not supported.")
        iaxes = []
        for ax in self.iaxes:
            ax_ = ax
            if ax > axis1:
                ax_ -= 1
            if ax > axis2:
                ax_ -= 1
            iaxes.append(ax_)

        iaxes = tuple(iaxes)
        return SparseNormal(self.v.trace(offset, axis1, axis2), iaxes)
    
    @staticmethod
    def concatenate(arrays, axis):
         # TODO: account for the case when axis is negative -----------------------------
        arrays = [assparsenormal(ar) for ar in arrays]
        iaxes = _validate_iaxes(arrays)
        if axis in iaxes:
            raise ValueError("Concatenation along independence axes "
                             "is not allowed.")

        v = Normal.concatenate([x.v for x in arrays], axis=axis)
        return SparseNormal(v, iaxes)
    
    @staticmethod
    def stack(arrays, axis):
        # TODO: account for the case when axis is negative -----------------------------
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

    def iid_copy(self):
        return SparseNormal(self.v.iid_copy(), self.iaxes)

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


def _validate_iaxes(seq):
    """Checks that the independence axes of the sparse normal arrays in `seq`
    are compatible, and returns those axes for the final broadcasted shape."""

    def reverse_axes(ndim, axes):
        """Converts a sequance of positive axes numbers into a sequence 
        of negative axes numbers.
        
        Examles:
        >>> reverse_axes(4, (1, 2))
        (-3, -2)

        >>> reverse_axes(3, (0,))
        (-2,)
        """
        return tuple(ndim - ax for ax in axes)

    iaxs = set(reverse_axes(x.ndim, x.iaxes) for x in seq 
               if len(x.v.emap.elem) !=0)
    # Reversing is to account for broadcasting - the independence 
    # axes numbers counted from the beginning of the array may not be 
    # the same for broadcastable arrys. When counted from the end, 
    # the axes numbers must always be the same as broadcasting 
    # cannot add new independence axes. 
    
    ndim = max(x.ndim for x in seq)

    if len(iaxs) == 1:
        return reverse_axes(ndim, iaxs.pop())
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
    
    out_axs = []  # out_axs[i] is the number of the i-th input axis 
                  # in the indexing result
    idx = []  # idx[i] is the index used for the i-th input axis.

    has_ellipsis = False
    new_dim = 0

    for k in key:
        if isinstance(k, int):
            idx.append(k)
            out_axs.append(None)
            continue

        if isinstance(k, slice):
            idx.append(k)
            out_axs.append(new_dim)
            new_dim += 1
            continue
        
        if k is Ellipsis:
            if has_ellipsis:
                raise IndexError("An index can only have a single "
                                 "ellipsis ('...').")
            
            has_ellipsis = True
            ep_in = len(out_axs)  # ellipsis position in the input
            ep_out = new_dim  # ellipsis position in the output
            continue

        if k is np.newaxis:
            new_dim += 1
            continue

        # Advanced indices (int and bool arrays) are not implemented. 

        raise IndexError("Only integers, slices (':'), ellipsis ('...'), "
                         "numpy.newaxis ('None') are valid indices.")
    
    # Checks if any input dimensions remain unconsumed.
    delta = x.ndim - len(out_axs)
    if delta > 0:
        if has_ellipsis:
            for i in range(ep_in, len(out_axs)):
                if out_axs[i] is not None:
                    out_axs[i] += delta

            out_axs[ep_in: ep_in] = range(ep_out, ep_out + delta)
            idx[ep_in: ep_in] = (slice(None, None, None) for _ in range(delta))
        else:
            out_axs.extend(range(new_dim, new_dim + delta))
            idx.extend(slice(None, None, None) for _ in range(delta))
    elif delta < 0:
        raise IndexError(f"Too many indices: the array is {x.ndim}-dimensional,"
                         f" but {len(out_axs)} indices were given.")
    
    iaxes = set(x.iaxes)
    for i, k in enumerate(idx):
        if i in iaxes and k != slice(None, None, None):
            raise IndexError("Only full slices (':') and ellipses ('...') "
                             "are valid indices for independence axes. "
                             f"Now the axis {i} is indexed with {k}.")
    return out_axs