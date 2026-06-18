import numpy as np

from numerics import sine_second_derivative_error


def test_operator_is_second_order():
    """Halving dx should cut the error ~4×, i.e. observed order ≈ 2."""
    dx_coarse, err_coarse = sine_second_derivative_error(100)
    dx_fine, err_fine = sine_second_derivative_error(200)
    order = np.log(err_coarse / err_fine) / np.log(dx_coarse / dx_fine)
    assert 1.9 < order < 2.1