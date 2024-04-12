from itertools import zip_longest
import numpy as np

from . import elementary


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
    
    def __getitem__(self, key):

        if isinstance(key, tuple) and Ellipsis in key:
            key_a = key + (slice(None),)
        else:
            key_a = key
        
        a_ = np.moveaxis(self.a, 0, -1)
        a = np.moveaxis(a_[key_a], -1, 0)  # TODO: clean this up
        return ElementaryMap(a, self.elem)

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
    
    def reshape(self, newvshape, **kwargs):
        newvshape = np.array(newvshape, ndmin=1)
        new_a = self.a.reshape((self.a.shape[0], *newvshape), **kwargs)
        return ElementaryMap(new_a, self.elem)
    
    def transpose(self, vaxes):
        if vaxes is None:
            vaxes = tuple(range(self.vndim))[::-1]
        new_a = self.a.transpose((0, *vaxes))
        return ElementaryMap(new_a, self.elem)
    
    def einsum(self, vsubs_seq, other):
        """ Computes einsum

        Args:
            vsubs_seq (Sequence[str]): parsed subscripts without ellipses, 
                (self_vsubs, other_subs, out_subs).
            other: constant array.
        """
        subs = f"...{vsubs_seq[0]},{vsubs_seq[1]}->...{vsubs_seq[2]}"
        new_a = np.einsum(subs, self.a, other)
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


def join(ops):  #TODO: remove in a future version

    # Stacks the operands along the 0th dimension

    if len(ops) > 2:
        union_elem = elementary.uunion(*[em.elem for em in ops])

        cat_a = np.zeros((len(union_elem), len(ops), *ops[0].vshape))
        for i, em in enumerate(ops):
            idx = [union_elem[k] for k in em.elem]
            cat_a[idx, i] = em.a

        return ElementaryMap(cat_a, union_elem)
    
    # The rest is an optimization for the case of two operands.
    op1, op2 = ops
    j1, j2 = 0, 1

    union_elem, swapped = elementary.ounion(op1.elem, op2.elem)

    if swapped:
        op1, op2 = op2, op1
        j1, j2 = j2, j1
    
    cat_a = np.zeros((len(union_elem), len(ops), *ops[0].vshape))
    cat_a[:len(op1.elem), j1] = op1.a
    idx = [union_elem[k] for k in op2.elem]
    cat_a[idx, j2] = op2.a

    return ElementaryMap(cat_a, union_elem)


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