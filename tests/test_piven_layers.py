import pytest
from piven.layers import Piven


def test_illegal_bound_values():
    with pytest.raises(ValueError):
        _ = Piven(3, -3)
