import numpy as np

from . import maps
from .random import Normal


def resolve(seq):
    cs = set([x.__class__ for x in seq if hasattr(x, "_mod")])
    if len(cs) == 0:
        return maps, Normal
    if len(cs) == 1:
        c = cs.pop()
        return c._mod, c
    
    raise TypeError(f"Classes {cs} cannot be combined in operations.")  # TODO: expand message


def fallback_to_normal(f):
    def f_(x, *args, **kwargs):
        if not hasattr(x, f.__name__):
            x = maps.lift(Normal, x)

        return f(x, *args, **kwargs)
        
    f_.__name__ = f.__name__
    return f_


# ---------- probability-related functions ----------


@fallback_to_normal
def iid_copy(x):
    """Creates an independent identically distributed copy of `x`."""
    return x.iid_copy()


@fallback_to_normal
def mean(x):
    """Expectation value, `<x>`."""
    return x.mean()


@fallback_to_normal
def var(x):
    """Variance, `<(x-<x>)(x-<x>)^*>`, where `*` denotes 
    complex conjugation, and `<...>` is the expectation value of `...`."""
    return x.var()


def cov(*args):
    """Covariance, generalizing `<outer((x-<x>), (y-<y>)^H)>`, 
    where `H` denotes conjugate transposition, and `<...>` is 
    the expectation value of `...`.
    
    Args:
        One or two random variables.

    Returns:
        For one random variable, `x`, the function returns `x.cov()`. 
        For two random variables, `x` and `y`, the function returns 
        their cross-covariance.
        
        The cross-covariance of two normal variables 
        is an array `c` with the dimension number equal 
        to the sum of the dimensions of `x` and `y`, whose components are
        `c[ijk... lmn...] = <(x[ijk..] - <x>)(y[lmn..] - <y>)*>`, 
        where the indices `ijk...` and `lmn...` run over the elements 
        of `x` and `y`, respectively, and `*` denotes complex conjugation.

        The cross-covariance of two sparse variables is an array 
        with the dimension number equal to the sum of the dense dimensions 
        of `x` and `y`, plus the number of their sparse (independence) 
        dimensions, which should be the same in `x` and `y`. 
        In the returned array, the regular dimensions 
        go first in the order they appear in the variable shapes, 
        and the independence dimensions are appended at the end.
        The resulting structure is the same as the structure produced 
        by repeated applications of `np.diagonal` over all the 
        independence dimensions of the full-sized covariance matrix `c` 
        for `x` and `y`.

    Raises:
        TypeError: 
            If the number of input arguments is not 1 or 2.

    Examples:
        Normal variables.

        >>> v1 = normal(size=(3,))  # shape (3,)
        >>> v2 = normal(size=(3,))  # shape (3,)
        >>> cov(v1, v1 + v2)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        >>> v = normal(size=(2, 3))
        >>> c = cov(v)
        >>> c.shape
        (2, 3, 2, 3)
        >>> np.all(c.reshape((v.size, v.size)) == np.eye(v.size))
        True

        >>> v1 = normal(size=2)
        >>> v2 = 0.5 * v1[0] + normal()
        >>> cov(v1, v2)
        array([0.5, 0. ])

        >>> v1 = normal(size=2)
        >>> v2 = 0.5 * v1[0] + normal(size=3)
        >>> cov(v1, v2)
        array([[0.5, 0.5, 0.5],
               [0. , 0. , 0. ]])

        >>> v1 = normal()
        >>> v2 = 1j * v1 + normal()
        >>> cov(v1, v2)
        array(0.-1.j)

        Sparse normal variables.

        >>> v1 = iid_repeat(normal(), 3)  # shape (3,)
        >>> v2 = iid_repeat(normal(), 3)  # shape (3,)
        >>> cov(v1, v1 + v2)
        array([1., 1., 1.])

        >>> v1 = iid_repeat(normal(size=3), 4)  # shape (4, 3)
        >>> v2 = iid_repeat(normal(size=2), 4)  # shape (4, 2)
        >>> cov(v1, v2).shape
        (3, 2, 4)
    """
    if len(args) == 0 or len(args) > 2:
        raise TypeError("The function can accept only one or two input "
                        f"arguments, while {len(args)} arguments are given.")
    
    mod, cls = resolve(args)
    return mod.cov(cls, *args)


# ---------- array functions ----------


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


@fallback_to_normal
def broadcast_to(x, shape):
    """Broadcasts the variable to a new shape."""
    return x.broadcast_to(shape)


def concatenate(arrays, axis=0):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to concatenate.")
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, a) for a in arrays]
    return mod.concatenate(cls, arrays, axis)


def stack(arrays, axis=0):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, a) for a in arrays]
    return mod.stack(cls, arrays, axis)


def hstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, a) for a in arrays]
    
    arrays = [a.ravel() if a.ndim == 0 else a for a in arrays]
    if arrays[0].ndim == 1:
        return mod.concatenate(cls, arrays, axis=0)
    
    return mod.concatenate(cls, arrays, axis=1)
    

def vstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, a) for a in arrays]

    if arrays[0].ndim <= 1:
        arrays = [a.reshape((1, -1)) for a in arrays]
    
    return mod.concatenate(cls, arrays, axis=0)


def dstack(arrays):
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, a) for a in arrays]

    if arrays[0].ndim <= 1:
        arrays = [a.reshape((1, -1, 1)) for a in arrays]
    elif arrays[0].ndim == 2:
        arrays = [a.reshape(a.shape + (1,)) for a in arrays]
    
    return mod.concatenate(cls, arrays, axis=2)


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
    mod, cls = resolve([op1, op2])
    return mod.bilinearfunc(cls, "dot", op1, op2)


def inner(op1, op2):
    mod, cls = resolve([op1, op2])
    return mod.bilinearfunc(cls, "inner", op1, op2)


def outer(op1, op2):
    mod, cls = resolve([op1, op2])
    return mod.bilinearfunc(cls, "outer", op1, op2)


def kron(op1, op2):
    mod, cls = resolve([op1, op2])
    return mod.bilinearfunc(cls, "kron", op1, op2)


def tensordot(op1, op2, axes=2):
    mod, cls = resolve([op1, op2])
    return mod.bilinearfunc(cls, "tensordot", op1, op2, [axes])


def einsum(subs, op1, op2):
    mod, cls = resolve([op1, op2])
    return mod.bilinearfunc(cls, "einsum", op1, op2, pargs=[subs])


# ---------- linear and linearized unary array ufuncs ----------


def linearized_unary(jmpf):
    if not jmpf.__name__.endswith("_jmp"):
        raise ValueError()
    
    f_name = jmpf.__name__[:-4]
    f = getattr(np, f_name)

    def flin(x):
        mod, cls = resolve([x])
        return mod.call_linearized(cls, x, f, jmpf)
    
    flin.__name__ = f_name
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