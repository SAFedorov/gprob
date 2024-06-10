import numpy as np
from numpy.exceptions import AxisError

from .normal_ import Normal, asnormal, print_normal, broadcast_to
from .external import einsubs


def iid_repeat(x, nrep=1, axis=0):
    """Creates a sparse array of independent identically distributed 
    copies of `x`."""

    x = assparsenormal(x)
    x = x.iid_copy()
    
    axis = _normalize_axis(axis, x.ndim + 1)
    isiax = x._isiax[:axis] + (True,) + x._isiax[axis:]
    v = x.v

    # Index that adds a new axis.
    idx = [slice(None, None, None)] * v.ndim
    idx.insert(axis, None)
    idx = tuple(idx)

    new_shape = v.shape[:axis] + (nrep,) + v.shape[axis:]
    v = broadcast_to(v[idx], new_shape)

    return SparseNormal(v, isiax)


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
    return SparseNormal(v, (False,) * v.ndim)


class SparseNormal:
    """Array of block-independent normal random variables."""

    __slots__ = ("v", "_isiax")
    __array_ufunc__ = None
    _normal_priority_ = 10
    
    def __init__(self, v, isiax):
        self.v = v
        self._isiax = isiax

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
        return SparseNormal(self.v.real, self._isiax)
    
    @property
    def imag(self):
        return SparseNormal(self.v.imag, self._isiax)
    
    @property
    def T(self):
        return self.transpose()
    
    @property
    def iscomplex(self):
        return self.v.iscomplex
    
    @property
    def iaxes(self):
        """Ordered sequence of axes along which the array elements 
        are independent from each other."""
        return tuple([i for i, b in enumerate(self._isiax) if b])
    
    def __repr__(self):
        return print_normal(self, extra_attrs=("iaxes",))
    
    def __len__(self):
        return len(self.v)
    
    def __neg__(self):
        return SparseNormal(-self.v, self._isiax)

    def __add__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v + other.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __radd__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v + self.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __sub__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v - other.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __rsub__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v - self.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __mul__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v * other.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __rmul__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v * self.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __truediv__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v / other.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __rtruediv__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v / self.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __pow__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = self.v ** other.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __rpow__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        v = other.v ** self.v
        isiax = _validate_iaxes([self, other])
        return SparseNormal(v, isiax)

    def __matmul__(self, other):
        if self.ndim == 0:
            raise ValueError("Matrix multiplication requires at least one "
                             "dimension, while the operand 1 has 0.")
        else:
            op_axis = self.ndim - 1

        if self._isiax[op_axis]:
            raise ValueError("Matrix multiplication affecting independence "
                             "axes is not supported. "
                             f"Axis {op_axis} of operand 1 is affected.")
        
        other = assparsenormal(other)

        if self.ndim == 1:
            isiax = (False,)  # == self._isiax, otherwise
                              # ValueError would have been thrown. 
        elif other.ndim <= max(self.ndim, 2):
            isiax = self._isiax
        else:
            # There are dimensions added by broadcasting.
            d = other.ndim - self.ndim
            isiax = (False,) * d + self._isiax

        # Calculates the contribution from the deterministic part of `other`.
        v = self.v @ other.mean()
        w = SparseNormal(v, isiax)
        
        if not _is_deterministic(other):
            # Adds the linearized contribution from the random part of `other`.
            # As `other`` is a sparse normal, the consistency of
            # independence axes is ensured by the addition operation.
            w += self.mean() @ (other - other.mean())

        return w

    def __rmatmul__(self, other):
        if self.ndim == 0:
            raise ValueError("Matrix multiplication requires at least one "
                             "dimension, while the operand 2 has 0.")
        elif self.ndim == 1:
            op_axis = 0
        else:
            op_axis = self.ndim - 2

        if self._isiax[op_axis]:
            raise ValueError("Matrix multiplication affecting independence "
                             "axes is not supported. "
                             f"Axis {op_axis} of operand 2 is affected.")
        
        other = assparsenormal(other)

        if other.ndim == 1:
            isiax = tuple([b for i, b in enumerate(self._isiax) 
                           if i != op_axis])
        elif other.ndim >= 2 and other.ndim <= self.ndim - 1:
            isiax = self._isiax
        else:
            d = other.ndim - self.ndim
            isiax = (False,) * d + self._isiax

        # Calculates the contribution from the deterministic part of `other`.
        v = other.mean() @ self.v
        w = SparseNormal(v, isiax)
        
        if not _is_deterministic(other):
            # Adds the linearized contribution from the random part of `other`.
            # As `other`` is a sparse normal, the consistency of
            # independence axes is ensured by the addition operation.
            w += (other - other.mean()) @ self.mean()

        return w
    
    def __getitem__(self, key):
        out_ax = _parse_index_key(self, key)
        isiax = tuple([self._isiax[ax] if ax is not None else False 
                       for ax in out_ax])
        return SparseNormal(self.v[key], isiax)
    
    def __setitem__(self, key, value):
        raise NotImplementedError
        
    # ---------- array methods ----------

    def conjugate(self):
        return SparseNormal(self.v.conjugate(), self._isiax)
    
    def conj(self):
        return self.conjugate()
    
    def cumsum(self, axis):
        if axis is None:
            # For regular arrays, None is the default value. For sparse arrays, 
            # it is not supported because these arrays usually 
            # cannot be flattened.
            raise ValueError("None value for the axis is not supported.")
        
        axis = _normalize_axis(axis, self.ndim)
        if self._isiax[axis]:
            raise ValueError("The computation of cumulative sums along "
                             "independence axes is not supported.")

        return SparseNormal(self.v.cumsum(axis), self._isiax)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        [axis1, axis2] = _normalize_axes([axis1, axis2], self.ndim)

        if self._isiax[axis1] or self._isiax[axis2]:
            raise ValueError("Taking diagonals along independence axes "
                             "is not supported.")
        s = {axis1, axis2}
        isiax = (tuple([b for i, b in enumerate(self._isiax) if i not in s]) 
                 + (False,))
        return SparseNormal(self.v.diagonal(offset, axis1, axis2), isiax)

    def flatten(self, order="C"):
        if not any(self._isiax) or self.ndim <= 1:
            return SparseNormal(self.v.flatten(order=order), self._isiax)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            isiax = (self._isiax[max_dim],)
            return SparseNormal(self.v.flatten(order=order), isiax)
        
        raise ValueError("Sparse normal variables can only be flattened if "
                         "all their dimensions except maybe one have size <=1, "
                         "or they have no independence axes.")
    
    def moveaxis(self, source, destination):
        source = _normalize_axis(source, self.ndim)
        destination = _normalize_axis(destination, self.ndim)
        
        isiax = list(self._isiax)
        isiax.insert(destination, isiax.pop(source))
        isiax = tuple(isiax)

        return SparseNormal(self.v.moveaxis(source, destination), isiax)
    
    def ravel(self, order="C"):
        if not any(self._isiax) or self.ndim <= 1:
            return SparseNormal(self.v.ravel(order=order), self._isiax)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            isiax = (self._isiax[max_dim],)
            return SparseNormal(self.v.ravel(order=order), isiax)
        
        raise ValueError("Sparse normal variables can only be flattened if "
                         "all their dimensions except maybe one have size <=1, "
                         "or they have no independence axes.")
    
    def reshape(self, newshape, order="C"):

        v = self.v.reshape(newshape, order)  
        # Reshaping the underlying varibale before the axes check is
        # to yield a meaningful error message if the shapes are inconsistent.

        newshape = v.shape  # To replace '-1' if it was in newshape originally.

        if v.size == 0:
            # The transformation of independence axes for zero-size variables 
            # cannot be determined unambiguously, so we always assume that 
            # the transformed variable has no independence axes.
            return SparseNormal(v, (False,) * v.ndim)

        new_dim = 0

        new_cnt = 1
        old_cnt = 1

        isiax = [False] * v.ndim
        for i, n in enumerate(self.shape):
            if self._isiax[i]:
                if n != 1:
                    # Skips all trivial dimensions first.
                    while new_dim < len(newshape) and newshape[new_dim] == 1:
                        new_dim += 1

                if (new_dim < len(newshape) and newshape[new_dim] == n 
                    and new_cnt == old_cnt):
                    
                    isiax[new_dim] = True
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
        
        isiax = tuple(isiax)
        return SparseNormal(v, isiax)
    
    def split(self, indices_or_sections, axis=0):
        axis = _normalize_axis(axis, self.ndim)

        if self._isiax[axis]:
            raise ValueError("Splitting along independence axes "
                             "is not supported.")
        
        vpieces = self.v.split(indices_or_sections, axis)
        return [SparseNormal(v, self._isiax) for v in vpieces]
    
    def sum(self, axis, keepdims=False):
        if not isinstance(axis, int) and not isinstance(axis, tuple):
            raise ValueError("`axis` must be an integer or "
                             "a tuple of integers.")
            # None, the default value for non-sparse arrays, is not supported,
            # because in the typical case the variable has at least one 
            # independence axis that cannot be contracted.

        sum_axes = _normalize_axes(axis, self.ndim)

        if any(self._isiax[ax] for ax in sum_axes):
            raise ValueError("The computation of sums along "
                             "independence axes is not supported.")
        
        if keepdims:
            isiax = self._isiax
        else:
            isiax = [b for i, b in enumerate(self._isiax) if i not in sum_axes]

        v = self.v.sum(axis=axis, keepdims=keepdims)
        return SparseNormal(v, isiax)

    def transpose(self, axes=None):
        if axes is None:
            isiax = self._isiax[::-1]
        else:
            isiax = tuple([self._isiax[ax] for ax in axes])

        return SparseNormal(self.v.transpose(axes), isiax)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        [axis1, axis2] = _normalize_axes([axis1, axis2], self.ndim)

        if self._isiax[axis1] or self._isiax[axis2]:
            raise ValueError("Traces along independence axes "
                             "are not supported.")
        s = {axis1, axis2}
        isiax = tuple([b for i, b in enumerate(self._isiax) if i not in s])
        return SparseNormal(self.v.trace(offset, axis1, axis2), isiax)
    
    @staticmethod
    def _concatenate(arrays, axis):
        arrays = [assparsenormal(ar) for ar in arrays]
        isiax = _validate_iaxes(arrays)
        axis = _normalize_axis(axis, len(isiax))
        
        if isiax[axis]:
            raise ValueError("Concatenation along independence axes "
                             "is not allowed.")

        v = Normal._concatenate([x.v for x in arrays], axis=axis)
        return SparseNormal(v, isiax)
    
    @staticmethod
    def _stack(arrays, axis):
        arrays = [assparsenormal(ar) for ar in arrays]
        isiax = _validate_iaxes(arrays)
        isiax = isiax[:axis] + (False,) + isiax[axis:]

        v = Normal._stack([x.v for x in arrays], axis=axis)
        return SparseNormal(v, isiax)
    
    # TODO: separate the individual functions - --------------------------------------

    @staticmethod
    def _bilinearfunc(name, op1, op2, *args, **kwargs):
        op1 = assparsenormal(op1)
        op2 = assparsenormal(op2)

        mn = f"_{name}_get_axes"

        op1, op2, op_axis1, op_axis2, isiax1, isiax2 = getattr(SparseNormal, mn)(op1, op2, *args, **kwargs)
        _validate_iaxes_bilinear(op1, op2, op_axis1, op_axis2)

        if not _is_deterministic(op1):
            v = Normal._bilinearfunc(name, op1.v, op2.v.mean(), *args, **kwargs)
            w = SparseNormal(v, isiax1)

            if not _is_deterministic(op2):
                w += SparseNormal._dot(op1.mean(), (op2 - op2.mean()))

        elif not _is_deterministic(op2):
            v = Normal._bilinearfunc(name, op1.v.mean(), op2.v, *args, **kwargs)
            w = SparseNormal(v, isiax2)

        else:
            w = getattr(np, name)(op1.v.mean(), op2.v.mean())

        return w
    
    @staticmethod
    def _dot_get_axes(op1, op2):
        if op1.ndim == 0:
            op_axis1 = -1
        elif op1.ndim == 1:
            op_axis1 = 0
        else:
            op_axis1 = op1.ndim - 1
        
        if op2.ndim == 0:
            op_axis2 = -1
        elif op2.ndim == 1:
            op_axis2 = 0
        else:
            op_axis2 = op2.ndim - 2

        isiax1 = op1._isiax

        isiax2 = list(op2._isiax)
        isiax2.pop(op2.ndim-1)
        isiax2 = (False,) * max(op1.ndim - 1, 0) + tuple(isiax2)

        return op1, op2, op_axis1, op_axis2, isiax1, isiax2

    @staticmethod
    def _inner_get_axes(op1, op2):
        op_axis1 = op1.ndim - 1
        op_axis2 = op2.ndim - 1
        
        isiax1 = op1._isiax[:-1]
        isiax2 = (False,) * (op1.ndim - 1) + op2._isiax[:-1]

        return op1, op2, op_axis1, op_axis2, isiax1, isiax2

    @staticmethod
    def _outer_get_axes(op1, op2):
        op1 = op1.ravel()
        op2 = op2.ravel()

        op_axis1 = None
        op_axis2 = None

        isiax1 = (True, False) if any(op1._isiax) else (False, False)
        isiax2 = (False, True) if any(op2._isiax) else (False, False)

        return op1, op2, op_axis1, op_axis2, isiax1, isiax2

    @staticmethod
    def _kron_get_axes(op1, op2):
        ndim = max(op1.ndim, op2.ndim)

        op_axis1 = None
        op_axis2 = None
        
        d1 = ndim - op1.ndim
        d2 = ndim - op2.ndim

        isiax1 = (False,) * d1 + op1._isiax
        isiax2 = (False,) * d2 + op2._isiax

        return op1, op2, op_axis1, op_axis2, isiax1, isiax2

    @staticmethod
    def _tensordot_get_axes(op1, op2, axes=2):
        try:
            iter(axes)
        except Exception:
            op_ax1 = list(range(op1.ndim - axes, op1.ndim))
            op_ax2 = list(range(0, axes))
        else:
            op_ax1, op_ax2 = axes
        # This is the same how numpy.tensordot handles the axes.
        # TODO: reuse this code from emaps module ------------------------------------------------

        isiax1 = tuple([b for i, b in enumerate(op1._isiax) if i not in op_ax1])
        isiax2 = tuple([b for i, b in enumerate(op2._isiax) if i not in op_ax2])
        isiax2 = (False,) * len(isiax1) + isiax2

        return op1, op2, op_ax1, op_ax2, isiax1, isiax2
    
    @staticmethod
    def _einsum(subs, op1, op2):
        def out_iaxes(op, insubs, outsubs):
            """Calculates the indices of the independence axes for
            the output operand."""
            
            for i, c in enumerate(insubs):
                if op._isiax[i] and c not in outsubs:
                    raise ValueError("Contraction over an independence"
                                     f" axis ({i}).")
                
            isiax = op._isiax + (False,)  # Augments with a default.  
            return tuple([op._isiax[outsubs.find(c)] for c in insubs])
        
        # TODO: convert to sparse normal or numerical operands
        # TODO: add a check that the indices of all independence axes appear in both operands

        # Converts the subscripts to an explicit form.
        (insu1, insu2), outsu = einsubs.parse(subs, (op1.shape, op2.shape))
        subs = f"{insu1},{insu2}->{outsu}"

        if isinstance(op1, SparseNormal) and isinstance(op2, SparseNormal):
            return (SparseNormal._einsum(subs, op1, op1.v.b)
                    + SparseNormal._einsum(subs, op1.v.b, op1))

        if isinstance(op1, SparseNormal) and not isinstance(op2, SparseNormal):
            isiax = out_iaxes(op1, insu1, outsu)
            v = Normal._einsum(subs, op1.v, op2)
            return SparseNormal(v, isiax)

        if isinstance(op2, SparseNormal) and not isinstance(op1, SparseNormal):
            isiax = out_iaxes(op2, insu2, outsu)
            v = Normal._einsum(subs, op1, op2.v)
            return SparseNormal(v, isiax)
        
        # The default that is not supposed to be reached, usually.
        return Normal._einsum(subs, op1, op2)
    
    @staticmethod
    def _fftfunc(name, x, n, axis, norm):
        raise NotImplementedError

    @staticmethod
    def _fftfunc_n(name, x, s, axes, norm):
        raise NotImplementedError

    # ---------- probability-related methods ----------

    # TODO: add a test that the doc strings are in sync with those of Normal --------------------

    def iid_copy(self):
        return SparseNormal(self.v.iid_copy(), self._isiax)

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


def _is_deterministic(x):
    """True if the sparse normal `x` is a promoted numeric constant, 
    False if it is not."""
    return len(x.v.emap.elem) == 0


def _normalize_axis(axis, ndim):
    """Ensures that the axis index is positive and within the array dimension.

    Returns:
        Normalized axis index.
    
    Raises:
        AxisError
            If the axis is out of range.
    """

    axis = axis.__index__()

    if axis < -ndim or axis > ndim - 1:
        raise AxisError(f"Axis {axis} is out of bounds "
                        f"for an array of dimension {ndim}.")
    if axis < 0:
        axis = ndim + axis

    return axis


def _normalize_axes(axes, ndim):
    """Ensures that the axes indices are positive, within the array dimension,
    and without duplicates.

    Returns:
        Tuple of normalized axes indices.
    
    Raises:
        AxisError
            If one of the axes is out of range.
        ValueError
            If there are duplicates among the axes.
    """

    # Essentially repeats the code of numpy's normalize_axis_tuple,
    # which does not seem to be part of the public API.

    if type(axes) not in (tuple, list):
        try:
            axes = [axes.__index__()]
        except TypeError:
            pass

    axes = tuple([_normalize_axis(ax, ndim) for ax in axes])

    if len(set(axes)) != len(axes):
        raise ValueError('Axes cannot repeat.')
    
    return axes


def _validate_iaxes(seq):
    """Checks that the independence axes of the sparse normal arrays in `seq`
    are compatible and returns them.
    
    Returns:
        (iaxes, ndim) of the final shape for the broadcasted arrays.
    """

    ndim = max(x.ndim for x in seq)
    iaxs = set((False,) * (ndim - x.ndim) + x._isiax 
                for x in seq if not _is_deterministic(x))
    # Reversing is to account for broadcasting - the independence 
    # axes numbers counted from the beginning of the array may not be 
    # the same for broadcastable arrys. When counted from the end, 
    # the axes numbers must always be the same as broadcasting 
    # cannot add new independence axes. 

    if len(iaxs) == 1:
        return iaxs.pop()
    elif len(iaxs) == 0:
        return (False,) * ndim
    
    # len > 1
    raise ValueError("Only sparse normals with identical independence "
                     "axes can be combined in operations.")


def _parse_index_key(x, key):
    """Validates the key and calculates the numbers of the output axes 
    for each input axis."""

    if not isinstance(key, tuple):
        key = (key,)
    
    out_axs = []  # out_axs[i] is the number of the i-th output axis 
                  # in the input shape.
    idx = []  # idx[i] is the index used for the i-th input axis.

    has_ellipsis = False
    used_dim = 0

    for k in key:
        if isinstance(k, int):
            used_dim += 1
            continue

        if isinstance(k, slice):
            idx.append(k)
            out_axs.append(used_dim)
            used_dim += 1
            continue
        
        if k is Ellipsis:
            if has_ellipsis:
                raise IndexError("An index can only have a single "
                                 "ellipsis ('...').")
            
            has_ellipsis = True
            ep_in = used_dim       # ellipsis position in the input
            ep_out = len(out_axs)  # ellipsis position in the output
            continue

        if k is np.newaxis:
            idx.append(None)
            out_axs.append(None)
            continue

        # Advanced indices (int and bool arrays) are not implemented. 

        raise IndexError("Only integers, slices (':'), ellipsis ('...'), "
                         "numpy.newaxis ('None') are valid indices.")
    
    # Checks if any input dimensions remain unconsumed.
    delta = x.ndim - used_dim
    if delta > 0:
        if has_ellipsis:
            for i in range(ep_out, len(out_axs)):
                if out_axs[i] is not None:
                    out_axs[i] += delta

            out_axs[ep_out: ep_out] = range(ep_in, ep_in + delta)
            idx[ep_out: ep_out] = (slice(None, None, None) for _ in range(delta))
        else:
            out_axs.extend(range(used_dim, used_dim + delta))
            idx.extend(slice(None, None, None) for _ in range(delta))
    elif delta < 0:
        raise IndexError(f"Too many indices: the array is {x.ndim}-dimensional,"
                         f" but {used_dim} indices were given.")
    
    for i, b in enumerate(x._isiax):
        if i not in out_axs:
            oi = -1
        else:
            oi = out_axs.index(i)
        
        if b and (oi == -1 or idx[oi] != slice(None, None, None)):
            raise IndexError("Only full slices (':') and ellipses ('...') "
                             "are valid indices for independence axes. "
                             f"Axis {i} is indexed with an invalid key.")
    return out_axs


def _validate_iaxes_bilinear(op1, op2, op_axis1, op_axis2):
    if not isinstance(op_axis1, tuple):
        op_axis1 = (op_axis1,)

    if not isinstance(op_axis2, tuple):
        op_axis2 = (op_axis2,)

    if any([op1._isiax[ax] for ax in op_axis1]):
        raise ValueError("Bilinear operations affecting "
                         "independence axes are not supported. "
                         f"Axes {op_axis1} of operand 1 are affected.")
    
    if any([op2._isiax[ax] for ax in op_axis2]):
        raise ValueError("Bilinear operations affecting "
                         "independence axes are not supported. "
                         f"Axes {op_axis2} of operand 2 are affected.")