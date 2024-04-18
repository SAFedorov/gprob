import numpy as np
from .normal import Normal


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
