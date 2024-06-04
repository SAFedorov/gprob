from gprob.normal_ import normal
from gprob.sparse import _parse_index_key, iid_repeat


def test_index_key_parsing():
    pass


def test_broadcasting():

    # Combine variables with compatible shapes and independence axes 
    # but different numbers of dimensions.
    v1 = iid_repeat(normal(size=(2,)), 3, axis=1)
    v2 = iid_repeat(normal() + 2., 3, axis=0)

    assert (v1 + v2).shape == (2, 3)
    assert (v1 + v2).iaxes == (1,)
    assert (v1 - v2).shape == (2, 3)
    assert (v1 - v2).iaxes == (1,)
    assert (v1 * v2).shape == (2, 3)
    assert (v1 * v2).iaxes == (1,)
    assert (v1 / v2).shape == (2, 3)
    assert (v1 / v2).iaxes == (1,)
    assert (v1 ** v2).shape == (2, 3)
    assert (v1 ** v2).iaxes == (1,)