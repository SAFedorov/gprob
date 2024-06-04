from itertools import zip_longest
from functools import reduce
from operator import mul

import numpy as np

from . import elementary
from .external import einsubs


class ElementaryMap:
    """Array of zero-mean normal random variables, represented as
    
    x[...] = sum_k a[i...] xi[i],
    
    where and `xi`s are elementary Gaussian variables that are independent and 
    identically-distributed, `xi`[i] ~ N(0, 1) for all i, 
    and ... is an arbitrary multi-dimensional index.
    """

    __slots__ = ("a", "elem", "vshape", "vndim")
    __array_ufunc__ = None

    def __init__(self, a, elem=None):

        if elem is None:
            elem = elementary.create(a.shape[0])
        elif len(elem) != a.shape[0]:
            raise ValueError(f"The length of the elementaries ({len(elem)}) "
                             "does not match the outer dimension of `a` "
                             f"({a.shape[0]}).")

        self.a = a
        self.vshape = a.shape[1:]
        self.vndim = a.ndim - 1
        self.elem = elem  # Dictionary of elementary variables {id -> k, ...}

    @property
    def a2d(self):
        vsz = reduce(mul, self.vshape, 1)
        return np.ascontiguousarray(self.a.reshape((self.a.shape[0], vsz)))

    def __neg__(self):
        return ElementaryMap(-self.a, self.elem)

    def __add__(self, other):
        """Adds two maps."""

        if self.elem is other.elem:
            # This is an optimization, primarily made to speed up in-place 
            # manipulations with array elements.

            new_a = (self.unsqueezed_a(other.vndim) 
                     + other.unsqueezed_a(self.vndim))
            return ElementaryMap(new_a, self.elem)
        
        if len(other.elem) == 0:
            # An optimization for the addition of empty maps, that emerge,
            # for example, when constants are lifted to normal variables.

            new_vshape = np.broadcast_shapes(self.vshape, other.vshape)
            return self.broadcast_to(new_vshape)
    
        op1, op2 = self, other
        union_elem, swapped = elementary.ounion(op1.elem, op2.elem)
        
        if swapped:
            op1, op2 = op2, op1

        dtype = np.result_type(self.a.dtype, other.a.dtype)    
        vsh = _broadcast_shapes(self.vshape, other.vshape)
        sum_a = np.zeros((len(union_elem), *vsh), dtype)
        sum_a[:len(op1.elem)] = op1.unsqueezed_a(len(vsh))
        idx = [union_elem[k] for k in op2.elem]
        sum_a[idx] += op2.unsqueezed_a(len(vsh))

        return ElementaryMap(sum_a, union_elem)

    def __mul__(self, other):
        """Multiplies the map by a scalar or array constant."""
        other = np.asanyarray(other)
        return ElementaryMap(self.unsqueezed_a(other.ndim) * other, self.elem)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divides the map by a scalar or array constant."""
        other = np.asanyarray(other)
        return ElementaryMap(self.unsqueezed_a(other.ndim) / other, self.elem)
    
    def __matmul__(self, other):
        """Matrix-multiplies the map by a constant array."""
        other = np.asanyarray(other)

        if self.vndim >= other.ndim:
            new_a = self.a @ other
        else:
            new_a = self.unsqueezed_a(other.ndim) @ other

            if self.vndim == 1:
                new_a = new_a.reshape(new_a.shape[:-2] + new_a.shape[-1:])

        return ElementaryMap(new_a, self.elem)

    def __rmatmul__(self, other):
        """Right matrix-multiplies the map by a constant array."""
        other = np.asanyarray(other)

        if self.vndim == 1:
            new_a = np.moveaxis(other @ self.a.T, -1, 0)
        elif self.vndim >= 2:
            new_a = other @ self.unsqueezed_a(other.ndim)
        else:
            raise ValueError("Scalars cannot be matrix-multiplied.")

        return ElementaryMap(new_a, self.elem)
    
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        
        key = (slice(None),) + key
        return ElementaryMap(self.a[key], self.elem)
    
    def __setitem__(self, key, value):
        """Sets `self` at the index `key` to `value`. 
        `value` must be an `ElementaryMap`. """

        if not isinstance(key, tuple):
            key = (key,)
        
        key = (slice(None),) + key

        if self.elem is not value.elem:
            self_, value = complete([self, value])
            self.a = self_.a
            self.elem = self_.elem
        
        idx_vdim = self.a[key].ndim - 1  # Determines the shape of the indexed 
                                         # section for broadcasting to work.  
        self.a[key] = value.unsqueezed_a(idx_vdim)

    def extend_to(self, new_elem):
        """Extends the map to a new list of elementary variables `new_elem` by
        adding zero entries. All the existing variables from `self.elem` must
        be present in `new_elem` (in an arbitrary order), 
        otherwise the function will fail."""
        
        new_shape = (len(new_elem), *self.a.shape[1:])
        new_a = np.zeros(new_shape, dtype=self.a.dtype)
        idx = [new_elem[k] for k in self.elem]
        new_a[idx] = self.a
        return ElementaryMap(new_a, new_elem)

    def pad_to(self, new_elem):
        """Extends the map to a new list of elementary variables `new_elem` by
        adding zero entries. This function assumes that `new_elem` contains 
        all the existing variables from `self.elem` in the same order
        in the beginning, and therefore the map should be just padded 
        with an appropriate number of zeros."""

        new_a = np.zeros((len(new_elem), *self.a.shape[1:]), dtype=self.a.dtype)
        new_a[:len(self.elem)] = self.a        
        return ElementaryMap(new_a, new_elem)
    
    def unsqueezed_a(self, vndim):
        """Enables broadcasting."""

        dn = vndim - self.vndim

        if dn > 0:
            sh = list(self.a.shape)
            sh[1:1] = (1,) * dn
            return self.a.reshape(sh)
        
        return self.a
    
    def broadcast_to(self, vshape):
        if self.vshape != vshape:
            new_a = self.unsqueezed_a(len(vshape))
            new_a = np.broadcast_to(new_a, (new_a.shape[0], *vshape))
            return ElementaryMap(new_a, self.elem)
            
        return self
    
    # ---------- array methods ----------
    
    def conj(self):
        return ElementaryMap(self.a.conj(), self.elem)
    
    @property
    def real(self):
        return ElementaryMap(self.a.real, self.elem)
    
    @property
    def imag(self):
        return ElementaryMap(self.a.imag, self.elem)
    
    def cumsum(self, vaxis=None, **kwargs):
        if vaxis is None:
            a = self.a.reshape((self.a.shape[0], -1))
            axis = 1
        else:
            a = np.atleast_2d(self.a)
            axis, = vax_to_ax([vaxis])

        new_a = np.cumsum(a, axis=axis, **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def diagonal(self, offset=0, vaxis1=0, vaxis2=1):
        axis1, axis2 = vax_to_ax([vaxis1, vaxis2])
        new_a = self.a.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        return ElementaryMap(new_a, self.elem)
    
    def flatten(self, order="C"):
        valid_orders = ["C", "F"]
        if order not in valid_orders:
            raise ValueError(f"Only {valid_orders} orders are supported.")
         
        vsz = reduce(mul, self.vshape, 1)
        new_a = self.a.reshape((self.a.shape[0], vsz), order=order).copy()
        return ElementaryMap(new_a, self.elem)
    
    def moveaxis(self, vsource, vdestination):
        source, destination = vax_to_ax([vsource, vdestination])
        new_a = np.moveaxis(self.a, source, destination)
        return ElementaryMap(new_a, self.elem)
    
    def ravel(self, order="C"):
        """Flattens the map's `a` to a 2D contiguous array, for which the 
        variable dimension is 1."""
        if order == "C":
            return ElementaryMap(self.a2d, self.elem)
        elif order == "F":
            vsz = reduce(mul, self.vshape, 1)
            new_a = self.a.reshape((self.a.shape[0], vsz), order="F")
            return ElementaryMap(np.asfortranarray(new_a), self.elem)

        raise ValueError("Only C and F orders are supported.")
    
    def reshape(self, newvshape, **kwargs):
        try:
            new_shape = (self.a.shape[0], *newvshape)
        except TypeError:
            new_shape = (self.a.shape[0], newvshape)

        new_a = self.a.reshape(new_shape, **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def sum(self, vaxis=None, **kwargs):
        if self.vndim == 0 or vaxis is None:
            vaxis = tuple(range(self.vndim))
        elif not isinstance(vaxis, tuple):
            vaxis = (vaxis,)

        axis = tuple(vax_to_ax(vaxis))  # sum needs axis to be a tuple not list
        new_a = self.a.sum(axis, **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def transpose(self, vaxes=None):
        if vaxes is None:
            vaxes = tuple(range(self.vndim))[::-1]

        axes = vax_to_ax(vaxes)
        new_a = self.a.transpose((0, *axes))
        return ElementaryMap(new_a, self.elem)
    
    def trace(self, offset=0, vaxis1=0, vaxis2=1, **kwargs):
        axis1, axis2 = vax_to_ax([vaxis1, vaxis2])
        new_a = self.a.trace(offset=offset, axis1=axis1, axis2=axis2, **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def split(self, ios, vaxis=0):
        axis, = vax_to_ax([vaxis])
        new_as = np.split(self.a, ios, axis=axis)
        return [ElementaryMap(a, self.elem) for a in new_as]
    
    def fftfunc(self, name, n, vaxis, norm):
        axis_a, = vax_to_ax([vaxis])
        func = getattr(np.fft, name)
        new_a = func(self.a, n, axis_a, norm)
        return ElementaryMap(new_a, self.elem)

    def fftfunc_n(self, name, s, vaxes, norm):
        if vaxes is None:
            vaxes = list(range(self.vndim))

        axes_a = vax_to_ax(vaxes)
        func = getattr(np.fft, name)
        new_a = func(self.a, s, axes_a, norm)
        return ElementaryMap(new_a, self.elem)
    
    # ---------- bilinear functions with numeric arrays ----------
    
    def einsum(self, vsubs, other, otherfirst=False):
        if not otherfirst:
            (ssu, osu), outsu = einsubs.parse(vsubs, (self.vshape, other.shape))
        else:
            (osu, ssu), outsu = einsubs.parse(vsubs, (other.shape, self.vshape))

        subs = f"...{ssu},{osu}->...{outsu}"

        new_a = np.einsum(subs, self.a, other)
        return ElementaryMap(new_a, self.elem)
    
    def inner(self, other, otherfirst=False):
        other = np.asanyarray(other)

        if other.ndim == 0:
            return ElementaryMap(self.a * other, self.elem)
        
        if self.vndim == 0:
            new_a = np.einsum("i, ... -> i...", self.a, other)
            return ElementaryMap(new_a, self.elem)

        if not otherfirst:
            new_a = np.inner(self.a, other)
        else:
            new_a = np.moveaxis(np.inner(other, self.a), other.ndim - 1, 0)

        return ElementaryMap(new_a, self.elem)
    
    def dot(self, other, otherfirst=False):
        other = np.asanyarray(other)

        if other.ndim == 0:
            return ElementaryMap(self.a * other, self.elem)
        
        if self.vndim == 0:
            new_a = np.einsum("i, ... -> i...", self.a, other)
            return ElementaryMap(new_a, self.elem)
        
        if not otherfirst:
            new_a = np.dot(self.a, other)
        else:
            if self.vndim == 1:
                new_a = np.einsum("...j, ij -> i...", other, self.a)
            else:
                new_a = np.moveaxis(np.dot(other, self.a), other.ndim - 1, 0)

        return ElementaryMap(new_a, self.elem)
    
    def outer(self, other, otherfirst=False):
        other = np.ravel(other)

        if not otherfirst:
            new_a = np.einsum("ij, k -> ijk", self.a2d, other)
        else:
            new_a = np.einsum("k, ij -> ikj", other, self.a2d)

        return ElementaryMap(new_a, self.elem)
    
    def kron(self, other, otherfirst=False):
        other = np.asanyarray(other)

        if not otherfirst:
            new_a = np.kron(self.unsqueezed_a(other.ndim), other)
        else:
            new_a = np.kron(other, self.unsqueezed_a(other.ndim))

        return ElementaryMap(new_a, self.elem)
    
    def tensordot(self, other, otherfirst=False, axes=2):
        try:
            iter(axes)
        except Exception:
            axes1 = list(range(-axes, 0))
            axes2 = list(range(0, axes))
        else:
            axes1, axes2 = axes
        # This is the same how numpy.tensordot handles the axes.

        if not otherfirst:
            axes1 = vax_to_ax(axes1)
            new_a = np.tensordot(self.a, other, axes=(axes1, axes2))
        else:
            axes2 = vax_to_ax(axes2)
            new_a = np.tensordot(other, self.a, axes=(axes1, axes2))
            new_a = np.moveaxis(new_a, -self.vndim - 1 + len(axes2), 0)

        return ElementaryMap(new_a, self.elem)


def vax_to_ax(vax):
    return [a + 1 if a >= 0 else a for a in vax]


def complete(ops):
    """Extends the maps to the union of their elementary variables."""

    if len(ops) == 1:
        return ops
    
    if len(ops) > 2:
        union_elem = elementary.uunion(*[x.elem for x in ops])
        return tuple(op.extend_to(union_elem) for op in ops)

    # The rest is an optimization for the case of two operands.
    op1, op2 = ops

    if op1.elem is op2.elem:
        return ops
    
    union_elem, swapped = elementary.ounion(op1.elem, op2.elem)

    if swapped:
        return op1.extend_to(union_elem), op2.pad_to(union_elem)
    
    return op1.pad_to(union_elem), op2.extend_to(union_elem)


def _broadcast_shapes(shape1, shape2):
    shapesit = zip_longest(reversed(shape1), reversed(shape2), fillvalue=0)
    return tuple(max(*d) for d in shapesit)[::-1]


def concatenate(emaps, vaxis=0):
    if len(emaps) == 1:
        return emaps[0]

    if vaxis < 0:
        vaxis = emaps[0].vndim + vaxis

    dims = [em.a.shape[vaxis + 1] for em in emaps]

    dtype = np.result_type(*[em.a.dtype for em in emaps])
    new_vshape = list(emaps[0].vshape)
    new_vshape[vaxis] = sum(dims)

    base_jidx = (slice(None),) * vaxis

    if len(emaps) > 2:
        union_elem = elementary.uunion(*[em.elem for em in emaps])
        cat_a = np.zeros((len(union_elem), *new_vshape), dtype)
        n1 = 0
        for i, em in enumerate(emaps):
            idx = [union_elem[k] for k in em.elem]
            n2 = n1 + dims[i]
            cat_a[idx, *base_jidx, n1: n2] = em.a
            n1 = n2

        return ElementaryMap(cat_a, union_elem)
    
    # The rest is an optimization for the case of two operands.
    op1, op2 = emaps

    # Indices along the variable dimension.
    jidx1 = base_jidx + (slice(None, dims[0]),)
    jidx2 = base_jidx + (slice(-dims[1], None),)

    union_elem, swapped = elementary.ounion(op1.elem, op2.elem)

    if swapped:
        op1, op2 = op2, op1
        jidx1, jidx2 = jidx2, jidx1
    
    cat_a = np.zeros((len(union_elem), *new_vshape), dtype)
    cat_a[:len(op1.elem), *jidx1] = op1.a
    idx = [union_elem[k] for k in op2.elem]
    cat_a[idx, *jidx2] = op2.a

    return ElementaryMap(cat_a, union_elem)


def stack(emaps, vaxis=0):
    # Essentially a copy of `concatenate`, with slightly less overhead.

    if vaxis < 0:
        vaxis = emaps[0].vndim + vaxis + 1

    base_jidx = (slice(None),) * vaxis

    if len(emaps) == 1:
        return emaps[0][*base_jidx, None]

    dtype = np.result_type(*[em.a.dtype for em in emaps])
    new_vshape = list(emaps[0].vshape)
    new_vshape.insert(vaxis, len(emaps)) 

    if len(emaps) > 2:
        union_elem = elementary.uunion(*[em.elem for em in emaps])

        cat_a = np.zeros((len(union_elem), *new_vshape), dtype)
        for i, em in enumerate(emaps):
            idx = [union_elem[k] for k in em.elem]
            cat_a[idx, *base_jidx, i] = em.a

        return ElementaryMap(cat_a, union_elem)
    
    # The rest is an optimization for the case of two operands.
    op1, op2 = emaps
    j1, j2 = 0, 1

    union_elem, swapped = elementary.ounion(op1.elem, op2.elem)

    if swapped:
        op1, op2 = op2, op1
        j1, j2 = j2, j1
    
    cat_a = np.zeros((len(union_elem), *new_vshape), dtype)
    cat_a[:len(op1.elem), *base_jidx, j1] = op1.a
    idx = [union_elem[k] for k in op2.elem]
    cat_a[idx, *base_jidx, j2] = op2.a

    return ElementaryMap(cat_a, union_elem)