from .normal_ import get_highest_class


def fft(x, n=None, axis=-1, norm=None):
    return _fftfunc("fft", x, n, axis, norm)


def ifft(x, n=None, axis=-1, norm=None):
    return _fftfunc("ifft", x, n, axis, norm)


def rfft(x, n=None, axis=-1, norm=None):
    return _fftfunc("rfft", x, n, axis, norm)


def irfft(x, n=None, axis=-1, norm=None):
    return _fftfunc("irfft", x, n, axis, norm)


def hfft(x, n=None, axis=-1, norm=None):
    return _fftfunc("hfft", x, n, axis, norm)


def ihfft(x, n=None, axis=-1, norm=None):
    return _fftfunc("ihfft", x, n, axis, norm)


def _fftfunc(name, x, n, axis, norm):
    cls = get_highest_class(x)
    return cls.fftfunc(name, x, n, axis, norm)


def fft2(x, s=None, axes=(-2, -1), norm=None):
    return _fftfunc_n("fft2", x, s, axes, norm)


def ifft2(x, s=None, axes=(-2, -1), norm=None):
    return _fftfunc_n("ifft2", x, s, axes, norm)


def fftn(x, s=None, axes=None, norm=None):
    return _fftfunc_n("fftn", x, s, axes, norm)


def ifftn(x, s=None, axes=None, norm=None):
    return _fftfunc_n("ifftn", x, s, axes, norm)


def rfft2(x, s=None, axes=(-2, -1), norm=None):
    return _fftfunc_n("rfft2", x, s, axes, norm)


def irfft2(x, s=None, axes=(-2, -1), norm=None):
    return _fftfunc_n("irfft2", x, s, axes, norm)


def rfftn(x, s=None, axes=None, norm=None):
    return _fftfunc_n("rfftn", x, s, axes, norm)


def irfftn(x, s=None, axes=None, norm=None):
    return _fftfunc_n("irfftn", x, s, axes, norm)


def _fftfunc_n(name, x, s, axes, norm):
    cls = get_highest_class(x)
    return cls.fftfunc_n(name, x, s, axes, norm)
