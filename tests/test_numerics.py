import pytest
from numerics import second_derivative
import numpy as np

def test_second_derivative_of_sine():
    x = np.linspace(0,2*np.pi,200)
    dx = x[1]-x[0]
    numeric = second_derivative(np.sin(x),dx)
    exact = -np.sin(x)
    assert np.allclose(numeric[1:-1], exact[1:-1], atol = 1e-3)

def test_second_derivative_exact_for_parabola():
    """f = x² → f'' = 2 everywhere, and central differences are
    *exact* for quadratics, so this matches to machine precision."""
    x = np.linspace(-1, 1, 50)
    dx = x[1] - x[0]
    d2 = second_derivative(x**2, dx)
    assert np.allclose(d2[1:-1], 2.0)