import numpy as np

from numerics import second_derivative


def heat_step(u: np.ndarray, alpha: float, dt: float, dx: float) -> np.ndarray:
    """Avance l'équation de la chaleur 1D d'un pas de temps (schéma FTCS).

    u_i^{n+1} = u_i^n + alpha*dt * (dérivée seconde de u en i).
    Bords de Dirichlet : les deux extrémités gardent leur valeur.
    """
    u_new = u.copy()
    # TROU 1 : calcule la dérivée seconde spatiale de u (réutilise ton opérateur)
    d2 = second_derivative(u,dx)
    # TROU 2 : mets à jour SEULEMENT les points intérieurs avec la formule FTCS
    u_new[1:-1] = u[1:-1]+alpha*d2[1:-1]*dt
    # les bords u_new[0] et u_new[-1] gardent leur valeur → rien à faire
    return u_new


def solve_heat(u0: np.ndarray, alpha: float, dx: float, dt: float, n_steps: int) -> np.ndarray:
    """Lance n_steps pas du solveur à partir du profil initial u0."""
    # TROU 3 : calcule le nombre de diffusion r, et refuse un pas de temps instable
    r = alpha*dt/dx**2
    if r > 0.5:
        raise ValueError(f"Instable : nombre de diffusion r={r:.3f} au-dessus de la limite")
    u = u0.copy()
    for _ in range(n_steps):
        u = heat_step(u, alpha, dt, dx)
    return u


def analytical_decaying_mode(x: np.ndarray, t: float, alpha: float, k: float) -> np.ndarray:
    """Solution exacte u(x,t) = sin(k x) * exp(-alpha k^2 t)."""
    # TROU 4 : renvoie le mode sinusoïdal qui décroît
    return (np.sin(k*x)*np.exp(-alpha*k**2*t))