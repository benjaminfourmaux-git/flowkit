from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numerics import sine_second_derivative_error


def run_convergence_study(resolutions):
    """Print a table of (n, dx, error, observed order) and return arrays."""
    dxs, errors = [], []
    print(f"{'n':>6} {'dx':>14} {'max_error':>14} {'order':>8}")
    prev = None
    for n in resolutions:
        dx, err = sine_second_derivative_error(n)
        if prev is None:
            order = "-"
        else:
            order = f"{np.log(prev[1] / err) / np.log(prev[0] / dx):.3f}"
        print(f"{n:>6} {dx:>14.4e} {err:>14.4e} {order:>8}")
        dxs.append(dx)
        errors.append(err)
        prev = (dx, err)
    return np.array(dxs), np.array(errors)


def plot_convergence(dxs, errors, path="docs/convergence.png"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dxs, errors, "o-", label="measured error")
    reference = errors[0] * (dxs / dxs[0]) ** 2
    ax.loglog(dxs, reference, "k--", label="slope = 2 (theory)")
    ax.set_xlabel("grid spacing  dx")
    ax.set_ylabel("max error")
    ax.set_title("Convergence of the second-derivative operator")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    dxs, errors = run_convergence_study([50, 100, 200, 400, 800])
    plot_convergence(dxs, errors)