import gprob as gp


def test_module_imports():
    names = ["fft", "linalg", "func", "maps", "normal_", "sparse"]

    for name in names:
        assert hasattr(gp, name)