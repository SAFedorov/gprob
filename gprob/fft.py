from .arrayops import resolve


def fft(x, n=None, axis=-1, norm=None):
    """One-dimensional discrete Fourier Transform of ``x``.
    
    See `numpy.fft.fft` for more details.
    """
    return _fftfunc("fft", x, n, axis, norm)


def ifft(x, n=None, axis=-1, norm=None):
    """One-dimensional inverse discrete Fourier Transform of ``x``.
    
    See `numpy.fft.ifft` for more details.
    """
    return _fftfunc("ifft", x, n, axis, norm)


def rfft(x, n=None, axis=-1, norm=None):
    """One-dimensional discrete Fourier Transform of a real-valued random 
    variable ``x``.
    
    See `numpy.fft.rfft` for more details.
    """
    return _fftfunc("rfft", x, n, axis, norm)


def irfft(x, n=None, axis=-1, norm=None):
    """Inverse of `rfft`.
    
    See `numpy.fft.irfft` for more details.
    """
    return _fftfunc("irfft", x, n, axis, norm)


def hfft(x, n=None, axis=-1, norm=None):
    """One-dimensional discrete Fourier Transform of a hermitian-valued random 
    variable ``x`` (i.e. a variable that has a real spectrum).
    
    See `numpy.fft.hfft` for more details.
    """
    return _fftfunc("hfft", x, n, axis, norm)


def ihfft(x, n=None, axis=-1, norm=None):
    """Inverse of `hfft`.
    
    See `numpy.fft.ihfft` for more details.
    """
    return _fftfunc("ihfft", x, n, axis, norm)


def _fftfunc(name, x, n, axis, norm):
    mod, cls = resolve([x])
    x = cls._mod.lift(cls, x)
    return mod.fftfunc(cls, name, x, n, axis, norm)


def fft2(x, s=None, axes=(-2, -1), norm=None):
    """Two-dimensional discrete Fourier Transform of ``x``.
    
    See `numpy.fft.fft2` for more details.
    """
    return _fftfunc_n("fft2", x, s, axes, norm)


def ifft2(x, s=None, axes=(-2, -1), norm=None):
    """Two-dimensional inverse discrete Fourier Transform of ``x``.
    
    See `numpy.fft.ifft2` for more details.
    """
    return _fftfunc_n("ifft2", x, s, axes, norm)


def rfft2(x, s=None, axes=(-2, -1), norm=None):
    """Two-dimensional discrete Fourier Transform of a real-valued random 
    variable ``x``.
    
    See `numpy.fft.rfft2` for more details.
    """
    return _fftfunc_n("rfft2", x, s, axes, norm)


def irfft2(x, s=None, axes=(-2, -1), norm=None):
    """Inverse of `rfft2`.
    
    See `numpy.fft.irfft2` for more details.
    """
    return _fftfunc_n("irfft2", x, s, axes, norm)


def fftn(x, s=None, axes=None, norm=None):
    """n-dimensional discrete Fourier Transform of ``x``.
    
    See `numpy.fft.fftn` for more details.
    """
    return _fftfunc_n("fftn", x, s, axes, norm)


def ifftn(x, s=None, axes=None, norm=None):
    """n-dimensional inverse discrete Fourier Transform of ``x``.
    
    See `numpy.fft.ifftn` for more details.
    """
    return _fftfunc_n("ifftn", x, s, axes, norm)


def rfftn(x, s=None, axes=None, norm=None):
    """n-dimensional discrete Fourier Transform of a real-valued random 
    variable ``x``.
    
    See `numpy.fft.rfftn` for more details.
    """
    return _fftfunc_n("rfftn", x, s, axes, norm)


def irfftn(x, s=None, axes=None, norm=None):
    """Inverse of `rfftn`.
    
    See `numpy.fft.irfftn` for more details.
    """
    return _fftfunc_n("irfftn", x, s, axes, norm)


def _fftfunc_n(name, x, s, axes, norm):
    mod, cls = resolve([x])
    x = cls._mod.lift(cls, x)
    return mod.fftfunc_n(cls, name, x, s, axes, norm)