import pytest

from physics import reynold_number


def test_reynolds_eau_turbulent():
    """Eau, 2 m/s dans un tube de 5 cm → Re ≈ 1e5 (turbulent)."""
    re = reynold_number(density=1000.0, velocity=2.0, length=0.05, viscosity=1.0e-3)
    assert re == pytest.approx(100_000.0)


def test_reynolds_cas_unitaire():
    """ρ=v=L=μ=1 → Re = 1, un cas trivial calculable de tête."""
    assert reynold_number(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)