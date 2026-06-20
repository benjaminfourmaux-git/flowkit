from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from heat import heat_step, analytical_decaying_mode


# --- parametres ---
L = 2 * np.pi
x = np.linspace(0.0, L, 41)
dx = x[1] - x[0]
alpha, k = 0.25, 1.0
dt = 0.4 * dx**2 / alpha        # r = 0.4 -> stable
n_frames = 180


# --- 1) construire l'historique : un profil sauvegarde par pas de temps ---
u = np.sin(k * x)
history = [u.copy()]
for _ in range(n_frames):
    # TROU 1 : avance d'un pas avec heat_step, puis ajoute une COPIE de u a history
    u = heat_step(u,alpha,dt,dx)
    history.append(u)


# --- 2) figure (fourni) ---
fig, ax = plt.subplots(figsize=(7, 4))
line_num, = ax.plot(x, history[0], "o-", color="#D85A30", lw=2, ms=4, label="solveur numerique")
line_exact, = ax.plot(x, history[0], "--", color="#378ADD", lw=1.5, label="solution exacte")
ax.set_xlim(0, L)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("temperature u")
ax.legend(loc="upper right")
ax.grid(True, ls=":")


def update(frame):
    t = frame * dt
    line_num.set_ydata(history[frame])
    line_exact.set_ydata(analytical_decaying_mode(x, t, alpha, k))
    ax.set_title(f"Diffusion de la chaleur - t = {t:.2f}")
    return line_num, line_exact


anim = FuncAnimation(fig, update, frames=len(history), interval=40, blit=False)

out = Path("docs/heat_diffusion.gif")
out.parent.mkdir(parents=True, exist_ok=True)
anim.save(out, writer=PillowWriter(fps=25))
print(f"saved {out}")
