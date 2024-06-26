import numpy as np

from . import normal_
from . import sparse

from .normal_ import Normal
from .sparse import SparseNormal


# ---------- probability-related functions ----------

def iid_copy(x):
    """Creates an independent identically distributed copy of `x`."""
    return x.iid_copy()


def cov():
    raise NotImplementedError # TODO: ------------------------------------------------------------------


# ---------- array functions ----------


def fallback_to_normal(func):
    def hedged_func(x, *args, **kwargs):
        try:
            return func(x, *args, **kwargs)
        except AttributeError:
            return func(normal_.asnormal(x), *args, **kwargs)
        
    hedged_func.__name__ = func.__name__
    return hedged_func


@fallback_to_normal
def diagonal(x, offset=0, axis1=0, axis2=1):
    return x.diagonal(offset=offset, axis1=axis1, axis2=axis2)


@fallback_to_normal
def sum(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


@fallback_to_normal
def cumsum(x, axis=None):
    return x.cumsum(axis=axis)


@fallback_to_normal
def moveaxis(x, source, destination):
    return x.moveaxis(source, destination)


@fallback_to_normal
def ravel(x, order="C"):
    return x.ravel(order=order)


@fallback_to_normal
def reshape(x, newshape, order="C"):
    """Gives a new shape to an array."""
    return x.reshape(newshape, order=order)


@fallback_to_normal
def transpose(x, axes=None):
    return x.transpose(axes=axes)


@fallback_to_normal
def trace(x, offset=0, axis1=0, axis2=1):
    return x.trace(offset=offset, axis1=axis1, axis2=axis2)


MODULE_DICT = {Normal: normal_, SparseNormal: sparse}


def resolve_module(seq):
    """Returns the module corresponding to the highest class 
    in the sequence `seq`. The classes are ordered according 
    to `_normal_priority_`, and the default is `Normal`."""

    if len(seq) == 1:
        return MODULE_DICT.get(seq[0].__class__, normal_)

    p = Normal._normal_priority_ - 1
    obj = max(seq, key=lambda a: getattr(a, "_normal_priority_", p), 
              default=None)
    
    return MODULE_DICT.get(obj.__class__, normal_)


def broadcast_to(x, shape):
    """Broadcasts the variable to a new shape."""
    mod = resolve_module([x])
    return mod.broadcast_to(x, shape)


def concatenate(arrays, axis=0):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to concatenate.")
    mod = resolve_module(arrays)
    return mod.concatenate(arrays, axis)


def stack(arrays, axis=0):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    mod = resolve_module(arrays)
    return mod.stack(arrays, axis)


def as_array_seq(arrays_or_scalars):
    return [x if hasattr(x, "ndim") else np.array(x) for x in arrays_or_scalars]


def hstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    arrays = as_array_seq(arrays)
    arrays = [a.ravel() if a.ndim == 0 else a for a in arrays]
    if arrays[0].ndim == 1:
        return concatenate(arrays, axis=0)
    
    return concatenate(arrays, axis=1)
    

def vstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    arrays = as_array_seq(arrays)
    if arrays[0].ndim <= 1:
        arrays = [a.reshape((1, -1)) for a in arrays]
    
    return concatenate(arrays, axis=0)


def dstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    arrays = as_array_seq(arrays)
    if arrays[0].ndim <= 1:
        arrays = [a.reshape((1, -1, 1)) for a in arrays]
    elif arrays[0].ndim == 2:
        arrays = [a.reshape((*a.shape, 1)) for a in arrays]
    
    return concatenate(arrays, axis=2)


@fallback_to_normal
def split(x, indices_or_sections, axis=0):   
    return x.split(indices_or_sections=indices_or_sections, axis=axis)


@fallback_to_normal
def hsplit(x, indices_or_sections):
    if x.ndim < 1:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    if x.ndim == 1:
        return split(x, indices_or_sections, axis=0)
    return split(x, indices_or_sections, axis=1)


@fallback_to_normal
def vsplit(x, indices_or_sections):
    if x.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(x, indices_or_sections, axis=0)


@fallback_to_normal
def dsplit(x, indices_or_sections):
    if x.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(x, indices_or_sections, axis=2)


#  ---------- bilinear functions ----------


def add(op1, op2):
    return op1 + op2


def subtract(op1, op2):
    return op1 - op2


def multiply(op1, op2):
    return op1 * op2


def divide(op1, op2):
    return op1 / op2


def power(op1, op2):
    return op1 ** op2


def matmul(op1, op2):
    return op1 @ op2


def dot(op1, op2):
    mod = resolve_module([op1, op2])
    return mod.dot(op1, op2)


def inner(op1, op2):
    mod = resolve_module([op1, op2])
    return mod.inner(op1, op2)


def outer(op1, op2):
    mod = resolve_module([op1, op2])
    return mod.outer(op1, op2)


def kron(op1, op2):
    mod = resolve_module([op1, op2])
    return mod.kron(op1, op2)


def tensordot(op1, op2, axes=2):
    mod = resolve_module([op1, op2])
    return mod.tensordot(op1, op2, axes)


def einsum(subs, op1, op2):
    mod = resolve_module([op1, op2])
    return mod.einsum(subs, op1, op2)


# ---------- linear and linearized unary array ufuncs ----------


def linearized_unary(jmpfunc):
    if not jmpfunc.__name__.endswith("_jmp"):
        raise ValueError()
    
    func_name = jmpfunc.__name__[:-4]
    func = getattr(np, func_name)

    def flin(x):
        mod = resolve_module([x])
        return mod.call_linearized(x, func, jmpfunc)
    
    flin.__name__ = func_name
    return flin


# Elementwise Jacobian-matrix products.
def exp_jmp(x, ans, a):     return a * ans
def exp2_jmp(x, ans, a):    return a * (ans * np.log(2.))
def log_jmp(x, ans, a):     return a / x
def log2_jmp(x, ans, a):    return a / (x * np.log(2.))
def log10_jmp(x, ans, a):   return a / (x * np.log(10.))
def sqrt_jmp(x, ans, a):    return a / (2. * ans)
def cbrt_jmp(x, ans, a):    return a / (3. * ans**2)
def sin_jmp(x, ans, a):     return a * np.cos(x)
def cos_jmp(x, ans, a):     return a * (-np.sin(x))
def tan_jmp(x, ans, a):     return a / np.cos(x)**2
def arcsin_jmp(x, ans, a):  return a / np.sqrt(1 - x**2)
def arccos_jmp(x, ans, a):  return a / (-np.sqrt(1 - x**2))
def arctan_jmp(x, ans, a):  return a / (1 + x**2)
def sinh_jmp(x, ans, a):    return a * np.cosh(x)
def cosh_jmp(x, ans, a):    return a * np.sinh(x)
def tanh_jmp(x, ans, a):    return a / np.cosh(x)**2
def arcsinh_jmp(x, ans, a): return a / np.sqrt(x**2 + 1)
def arccosh_jmp(x, ans, a): return a / np.sqrt(x**2 - 1)
def arctanh_jmp(x, ans, a): return a / (1 - x**2)
def conjugate_jmp(x, ans, a): return a.conj()


exp = linearized_unary(exp_jmp)
exp2 = linearized_unary(exp2_jmp)
log = linearized_unary(log_jmp)
log2 = linearized_unary(log2_jmp)
log10 = linearized_unary(log10_jmp)
sqrt = linearized_unary(sqrt_jmp)
cbrt = linearized_unary(cbrt_jmp)
sin = linearized_unary(sin_jmp)
cos = linearized_unary(cos_jmp)
tan = linearized_unary(tan_jmp)
arcsin = linearized_unary(arcsin_jmp)
arccos = linearized_unary(arccos_jmp)
arctan = linearized_unary(arctan_jmp)
sinh = linearized_unary(sinh_jmp)
cosh = linearized_unary(cosh_jmp)
tanh = linearized_unary(tanh_jmp)
arcsinh = linearized_unary(arcsinh_jmp)
arccosh = linearized_unary(arccosh_jmp)
arctanh = linearized_unary(arctanh_jmp)
conjugate = conj = linearized_unary(conjugate_jmp)