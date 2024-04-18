from itertools import zip_longest
import numpy as np

from . import elementary
from .external import einsubs


class ElementaryMap:
    """Array of zero-mean normal random variables, represented as
    
    x[...] = sum_k a[i...] xi[i],
    
    where and `xi`s are elementary Gaussian variables that are independent and 
    identically-distributed, `xi`[i] ~ N(0, 1) for all i, and ... is a 
    multi-dimensional index.
    """

    __slots__ = ("a", "elem", "vshape", "vndim")

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
        return np.ascontiguousarray(self.a.reshape((self.a.shape[0], -1)))

    def __neg__(self):
        return ElementaryMap(-self.a, self.elem)  # TODO: test how this performs

    def __add__(self, other):
        """Adds two maps."""

        # TODO: This is missing broadcasting, probably better to remove
        #if self.elem is other.elem:
        #    return ElementaryMap(self.a + other.a, self.elem)
    
        op1, op2 = self, other
        union_elem, swapped = elementary.ounion(op1.elem, op2.elem)
        
        if swapped:
            op1, op2 = op2, op1
            
        vsh = _broadcast_shapes(self.vshape, other.vshape)
        sum_a = np.zeros((len(union_elem), *vsh))
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
        new_a = self.unsqueezed_a(other.ndim - 1) @ other
        return ElementaryMap(new_a, self.elem)

    def __rmatmul__(self, other):
        """Right matrix-multiplies the map by a constant array."""
        other = np.asanyarray(other)

        if self.vndim == 1:
            new_a = np.moveaxis(other @ self.a.T, -1, 0)  # TODO: test this implementation for broadcasting
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

    def extend_to(self, new_elem):
        
        new_shape = (len(new_elem), *self.a.shape[1:])
        new_a = np.zeros(new_shape)
        idx = [new_elem[k] for k in self.elem]
        new_a[idx] = self.a

        return ElementaryMap(new_a, new_elem)

    def pad_to(self, new_elem):
        """Extends the map assuming that `new_elem` contain `self.elem` as
        their beginning."""

        new_a = np.zeros((len(new_elem), *self.a.shape[1:]))
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

        vndim = len(vshape)
        dn = vndim - self.vndim

        if dn > 0:
            new_a = self.unsqueezed_a(vndim)
            new_a = np.broadcast_to(new_a, (new_a.shape[0], *vshape))
            ElementaryMap(new_a, self.elem)
            
        return self
    
    # ---------- array methods ----------
    
    def conj(self):  # TODO: change to "conjugate" as numpy uses this name for the dispatch
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
            a = self.a
            if vaxis >= 0:
                axis = vaxis + 1
            else:
                axis = vaxis

        new_a = np.cumsum(a, axis=axis, **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def diagonal(self, offset=0, vaxis1=0, vaxis2=1):
        if vaxis1 >= 0:
            axis1 = vaxis1 + 1
        else:
            axis1 = vaxis1

        if vaxis2 >= 0:
            axis2 = vaxis2 + 1
        else:
            axis2 = vaxis2

        new_a = self.a.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        return ElementaryMap(new_a, self.elem)
    
    def flatten(self, **kwargs):
        # Flattens to 1d first to create a copy.
        new_a = self.a.flatten(**kwargs).reshape((len(self.elem), -1))
        return ElementaryMap(new_a, self.elem)
    
    def moveaxis(self, vsource, vdestination):
        if vsource >= 0:
            source = vsource + 1
        else:
            source = vsource

        if vdestination >= 0:
            destination = vdestination + 1
        else:
            destination = vdestination

        new_a = np.moveaxis(self.a, source, destination)
        return ElementaryMap(new_a, self.elem)
    
    def ravel(self):
        """Flattens the map's `a` to a 2D contiguous array, for which the 
        variable dimension is 1."""
        return ElementaryMap(self.a2d, self.elem)
    
    def reshape(self, newvshape, **kwargs):
        newvshape = np.array(newvshape, ndmin=1)
        new_a = self.a.reshape((self.a.shape[0], *newvshape), **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def sum(self, vaxis=None, **kwargs):
        if vaxis is None:
            vaxis = tuple(range(self.vndim))
        elif not isinstance(vaxis, tuple):
            vaxis = (vaxis,)

        axis = tuple(ax + 1 if ax >= 0 else ax for ax in vaxis)
        new_a = self.a.sum(axis, **kwargs)

        return ElementaryMap(new_a, self.elem)
    
    def transpose(self, vaxes=None):
        if vaxes is None:
            vaxes = tuple(range(1, self.vndim + 1))[::-1]
        new_a = self.a.transpose((0, *vaxes))
        return ElementaryMap(new_a, self.elem)
    
    def trace(self, offset=0, vaxis1=0, vaxis2=1, **kwargs): # Probably need to introduce a helper to do ax1, ax2 = from_vaxes(vax1, vax2)
        if vaxis1 >= 0:
            axis1 = vaxis1 + 1
        else:
            axis1 = vaxis1

        if vaxis2 >= 0:
            axis2 = vaxis2 + 1
        else:
            axis2 = vaxis2

        new_a = self.a.trace(offset=offset, axis1=axis1, axis2=axis2, **kwargs)
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

        # This is the same how numpy.tensordot handles the axes.
        try:
            iter(axes)
        except Exception:
            axes1 = list(range(-axes, 0))
            axes2 = list(range(0, axes))
        else:
            axes1, axes2 = axes

        if not otherfirst:
            axes1 = [a + 1 if a >= 0 else a for a in axes1]
            new_a = np.tensordot(self.a, other, axes=(axes1, axes2))
        else:
            axes2 = [a + 1 if a >= 0 else a for a in axes2]
            new_a = np.tensordot(other, self.a, axes=(axes1, axes2))
            new_a = np.moveaxis(new_a, -self.vndim + len(axes2), 0)

        return ElementaryMap(new_a, self.elem)


def complete(ops):
    """Completes the maps."""

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


def concatenate(emaps, vaxis=0, dtype=None):
    if not dtype:
        dtype = emaps[0].a.dtype

    if vaxis < 0:
        vaxis = emaps[0].vndim - 1 + vaxis

    dims = [em.a.shape[vaxis + 1] for em in emaps]

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


def stack(emaps, vaxis=0, dtype=None):

    # Essentially a copy of `concatenate`, with slightly less overhead.

    if not dtype:
        dtype = emaps[0].a.dtype

    if vaxis < 0:
        vaxis = emaps[0].vndim - 1 + vaxis

    base_jidx = (slice(None),) * vaxis

    if len(emaps) > 2:
        union_elem = elementary.uunion(*[em.elem for em in emaps])

        cat_a = np.zeros((len(union_elem), len(emaps), *emaps[0].vshape), dtype)
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
    
    cat_a = np.zeros((len(union_elem), len(emaps), *emaps[0].vshape), dtype)
    cat_a[:len(op1.elem), *base_jidx, j1] = op1.a
    idx = [union_elem[k] for k in op2.elem]
    cat_a[idx, *base_jidx, j2] = op2.a

    return ElementaryMap(cat_a, union_elem)


def hstack(emaps, dtype=None):
    if emaps[0].vndim == 0:
        return stack(emaps, vaxis=0, dtype=dtype)
    elif emaps[0].vndim == 1:
        return concatenate(emaps, vaxis=0, dtype=dtype)
    
    return concatenate(emaps, vaxis=1, dtype=dtype)
    

def vstack(emaps, dtype=None):
    if emaps[0].vndim <= 1:
        emaps = [em.reshape((1, -1)) for em in emaps]
    
    return concatenate(emaps, vaxis=0, dtype=dtype)


def dstack(emaps, dtype):
    if emaps[0].vndim <= 1:
        emaps = [em.reshape((1, -1, 1)) for em in emaps]
    elif emaps[0].vndim == 2:
        emaps = [em.reshape((*em.vshape, 1)) for em in emaps]
    
    return concatenate(emaps, vaxis=2, dtype=dtype)