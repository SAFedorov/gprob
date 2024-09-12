import pytest
from gprob.external import einsubs


def validate_index_list(sl):
    assert len(set(sl)) == len(sl)
    for s in sl:
        assert isinstance(s, str)
        assert len(s) == 1


def test_get_symbol():
    sl = [einsubs.get_symbol(k) for k in range(20)]
    validate_index_list(sl)

    sl = [einsubs.get_symbol(200 + k) for k in range(20)]
    validate_index_list(sl)


def test_gen_unused_symbols():
    sl = list(einsubs.gen_unused_symbols("", 20))
    validate_index_list(sl)

    used = "abd"
    sl = list(einsubs.gen_unused_symbols(used, 20))
    validate_index_list(sl)
    for u in used:
        assert u not in sl


def test_parse():
    [insu1, insu2], outsu = einsubs.parse("i, i -> i", ((3,), (3,)))
    assert insu1 == "i"
    assert insu2 == "i"
    assert outsu == "i"

    [insu1, insu2], outsu = einsubs.parse("i, i", ((3,), (3,)))
    assert insu1 == "i"
    assert insu2 == "i"
    assert outsu == ""

    [insu1, insu2], outsu = einsubs.parse("a, b", ((3,), (3,)))
    assert insu1 == "a"
    assert insu2 == "b"
    assert outsu == "ab"

    [insu1, insu2], outsu = einsubs.parse("... i, i -> ...", ((2, 3), (3,)))
    assert insu1 == outsu + "i"
    assert insu2 == "i"

    # Ellipsis for a scalar.
    [insu1, insu2], outsu = einsubs.parse("i, ... -> i...", ((3,), tuple()))
    assert insu1 == "i"
    assert insu2 == ""
    assert outsu == "i"

    # Implicit output.
    [insu1, insu2], outsu = einsubs.parse("i, ... i", ((3,), (2, 3)))
    assert insu1 == "i"
    assert insu2 == outsu + "i"

    # Abusive inputs.
    with pytest.raises(ValueError):
        einsubs.parse("i, i ->-> i", ((3,), (3,)))
    with pytest.raises(ValueError):
        einsubs.parse("i, i ->- i", ((3,), (3,)))
    with pytest.raises(ValueError):
        einsubs.parse("i, i >- i", ((3,), (3,)))
    with pytest.raises(ValueError):
        einsubs.parse("i, i > i", ((3,), (3,)))
    with pytest.raises(ValueError):
        einsubs.parse("i, i - i", ((3,), (3,)))
    with pytest.raises(ValueError):
        einsubs.parse("i, i <- i", ((3,), (3,)))

    with pytest.raises(ValueError):
        einsubs.parse(None, ((3,), (3,)))  # Subscripts not a string.

    with pytest.raises(ValueError):
        einsubs.parse(".. i, i -> ..", ((2, 3), (3,)))  # Invalid ellipses.

    with pytest.raises(ValueError):
        einsubs.parse("... ijk, i -> ...jk", ((2, 3), (3,)))  
        # Mismatching count.

    with pytest.raises(ValueError):
        einsubs.parse("i, i -> k", ((3,), (3,)))  
        # Output character that is not in the input.