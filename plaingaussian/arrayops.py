import numpy as np
from .normal import Normal, asnormal
from . import emaps


# ---------- linear and linearized array ufuncs ----------

def exp(x):
    new_b = np.exp(x.b)
    return Normal(x.emap * new_b, new_b)


def exp2(x):
    new_b = np.exp2(x.b)
    return Normal(x.emap * (new_b * np.log(2.)), new_b)


def log(x): return Normal(x.emap / x.b, np.log(x.b))
def log2(x): return Normal(x.emap / (x.b * np.log(2.)), np.log2(x.b))
def log10(x): return Normal(x.emap / (x.b * np.log(10.)), np.log10(x.b))


def sqrt(x):
    new_b = np.sqrt(x.b)
    return Normal(x.emap /(2. * new_b), new_b)


def cbrt(x):
    new_b = np.cbrt(x.b)
    return Normal(x.emap /(3. * new_b**2), new_b)


def sin(x): return Normal(x.emap * np.cos(x.b), np.sin(x.b))
def cos(x): return Normal(x.emap * (-np.sin(x.b)), np.cos(x.b))
def tan(x): return Normal(x.emap / np.cos(x.b)**2, np.tan(x.b))


def arcsin(x): return Normal(x.emap / np.sqrt(1 - x.b**2), np.arcsin(x.b))
def arccos(x): return Normal(x.emap / (-np.sqrt(1 - x.b**2)), np.arccos(x.b))
def arctan(x): return Normal(x.emap / (1 + x.b**2), np.arctan(x.b))


def sinh(x): return Normal(x.emap * np.cosh(x.b), np.sinh(x.b))
def cosh(x): return Normal(x.emap * np.sinh(x.b), np.cosh(x.b))
def tanh(x): return Normal(x.emap / np.cosh(x.b)**2, np.tanh(x.b))


def arcsinh(x): return Normal(x.emap / np.sqrt(x.b**2 + 1), np.arcsinh(x.b))
def arccosh(x): return Normal(x.emap / np.sqrt(x.b**2 - 1), np.arccosh(x.b))
def arctanh(x): return Normal(x.emap / (1 - x.b**2), np.arctanh(x.b))


def conjugate(x): return Normal(x.emap.conj(), x.b.conj())
def conj(x): return conjugate(x)  # In numpy, conjugate (not conj) is a ufunc


# ---------- linear array functions ----------


def sum(x, axis=None, dtype=None, keepdims=False):
    # "where" is absent because its broadcasting is not implemented.
    # "initial" is also not implemented.
    return x.sum(axis=axis, dtype=dtype, keepdims=keepdims)


def cumsum(x, axis=None, dtype=None):
    return x.cumsum(axis=axis, dtype=dtype)


def reshape(x, newshape, order="C"):
    return x.reshape(newshape, order=order)


def transpose(x, axes=None):
    return x.transpose(axes=axes)


def concatenate(arrays, axis=0, dtype=None):
    return _concatfunc("concatenate", arrays, axis, dtype=dtype)


def stack(arrays, axis=0, dtype=None):
    return _concatfunc("stack", arrays, axis, dtype=dtype)


def hstack(arrays, dtype=None):
    return _concatfunc("hstack", arrays, dtype=dtype)


def vstack(arrays, dtype=None):
    return _concatfunc("vstack", arrays, dtype=dtype)


def dstack(arrays, dtype=None):
    return _concatfunc("dstack", arrays, dtype=dtype)


# Function that applies to the concatenate mode family: concatenate, stack, hstack, vstack, dstack
def _concatfunc(name, arrays, *args, **kwargs):
    arrays = [asnormal(ar) for ar in arrays]
    
    if len(arrays) == 0:
        raise ValueError("Need at least one array.")
    elif len(arrays) == 1:
        return arrays[0]
    
    b = getattr(np, name)([x.b for x in arrays], *args, **kwargs)
    em = getattr(emaps, name)([x.emap for x in arrays], *args, **kwargs)

    return Normal(em, b)


# TODO: split family: split, hsplit, vsplit, dsplit

# TODO: linear algebra family: dot, matmul, einsum, inner, outer, kron
def einsum(subs, op1, op2):
    if isinstance(op2, Normal) and isinstance(op1, Normal):
        raise NotImplementedError("Einsums between two normal variables are not implemented.")

    if isinstance(op1, Normal) and not isinstance(op2, Normal):
        b = np.einsum(subs, op1.b, op2)
        em = op1.emap.einsum(subs, op2)
        return Normal(em, b)
    
    if isinstance(op2, Normal) and not isinstance(op1, Normal):
        b = np.einsum(subs, op1, op2.b)
        em = op2.emap.einsum(subs, op1, otherfirst=True)
        return Normal(em, b)

    return np.einsum(subs, op1, op2)