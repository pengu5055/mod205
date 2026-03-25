"""
Plot the results of mm-a-scan.py
"""
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from src import *

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

fn = f"./Data/mm_scan_a0.1_g100_size4_N600.h5"

results = load_scan(fn)
print(f"Loaded {len(results)} results from {fn}")
alpha_values = np.array([res["alpha"] for res in results])
omega_values = np.array([res["omega"] for res in results])
iters = np.array([res["iter"] for res in results])
Cs = np.array([res["C"] for res in results])

data = np.load("goofy-test.npz")
state = data["state"]
mask = data["mask"]
residuals = data["residuals"]
C = data["poiseuille_coeff"]

print("iterations:", iters)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

colors = cmr.take_cmap_colors("cmr.tropical", len(results), cmap_range=(0.0, 0.8))
cm = cmr.get_sub_cmap("cmr.tropical", 0.0, 0.8)
cm_r = cmr.get_sub_cmap("cmr.tropical_r", 0.2, 1.0)
for i, res in enumerate(results[::-1]):
    omega = res["omega"]
    residuals = res["residuals"]
    it = res["iter"]
    ax[0].plot(residuals, lw=2, color=colors[::-1][i])

norm = mpl.colors.Normalize(vmin=omega_values.min(), vmax=omega_values.max())
sm = plt.cm.ScalarMappable(cmap=cm_r, norm=norm)
cbar = plt.colorbar(sm, ax=ax[0])
cbar.set_label(r"$\omega$")

ax[0].set_yscale("log")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Residual")
ax[0].set_title("Convergence of Residuals")

cm = cmr.get_sub_cmap("cmr.chroma", 0.15, 0.75)
cm.set_bad(color="lightgray", alpha=0.0)
state /= np.max(state[mask])
state[state < 0.01] = np.nan

norm = mpl.colors.LogNorm(vmin=0.01, vmax=np.nanmax(state[mask]))
im = ax[1].imshow(state, cmap=cm, norm=norm, zorder=4)
ax[1].imshow(mask, cmap="gray", alpha=1.0, zorder=3)
ax[1].set_title(f"Steady-State Solution (C={C:.4f})")
fig.colorbar(im, ax=ax[1], label="Normalized Velocity")
ax[1].grid(False)
ax[1].axis("off")

plt.suptitle("SOR Convergence for Pipe with Mentor Profile")
plt.tight_layout()
plt.subplots_adjust(top=0.835)
parstr = f"Evaluated with: $N={600}$, $\\alpha_0 = {alpha_values.min():.2f}$, $\\alpha_1 = {alpha_values.max():.2f}$, $N_{{\\alpha}} = {len(alpha_values)}$ on {4} Chunks/MPI Ranks"
plt.figtext(0.5, 0.92, parstr, ha='center', va="center", transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.savefig("./Images/mm_a_scan.pdf", dpi=450)
plt.show()
