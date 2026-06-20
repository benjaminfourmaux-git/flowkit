import numpy as np
import pytest

from heat import solve_heat, analytical_decaying_mode


def test_heat_matches_analytical_decay():
    """Partant de sin(kx), le solveur doit suivre la décroissance exacte."""
    L = 2 * np.pi
    x = np.linspace(0.0, L, 101)
    dx = x[1] - x[0]
    alpha, k = 0.1, 1.0
    dt = 0.4 * dx**2 / alpha          # r = 0.4, donc stable
    n_steps = 80

    u0 = np.sin(k * x)
    u_num = solve_heat(u0, alpha, dx, dt, n_steps)

    t_final = n_steps * dt
    # TROU A : la solution exacte au meme instant t_final
    #          (appelle analytical_decaying_mode avec les bons arguments)
    u_exact = analytical_decaying_mode(x,t_final,alpha,k)
    assert np.allclose(u_num, u_exact, atol=1e-3)


def test_unstable_timestep_is_rejected():
    """Un pas de temps trop grand (r > 1/2) doit lever une erreur."""
    x = np.linspace(0.0, 1.0, 51)
    dx = x[1] - x[0]
    alpha = 0.1
    dt = 0.6 * dx**2 / alpha          # r = 0.6, instable
    u0 = np.sin(np.pi * x)
    # TROU B : complete pytest.raises(...) avec le type d'erreur attendu
    with pytest.raises(ValueError):
        solve_heat(u0, alpha, dx, dt, 5)
