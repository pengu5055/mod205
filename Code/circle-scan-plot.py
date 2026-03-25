"""
Plot results from circle-scan.
"""
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from src import *

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

N = 418
omega_ansatz = lambda alpha, N: 2 / (1 + (np.pi * alpha / N))
fn = "./Data/circle_scan_a0.1_g100_size49_N418.h5"

results = load_scan(fn)
print(f"Loaded {len(results)} results from {fn}")
alpha_values = np.array([res["alpha"] for res in results])
omega_values = np.array([res["omega"] for res in results])
iters = np.array([res["iter"] for res in results])
Cs = np.array([res["C"] for res in results])


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

colors = cmr.take_cmap_colors("cmr.tropical", len(results), cmap_range=(0.0, 0.8))
cm = cmr.get_sub_cmap("cmr.tropical", 0.0, 0.8)
cm_r = cmr.get_sub_cmap("cmr.tropical_r", 0.2, 1.0)
for i, res in enumerate(results):
    omega = res["omega"]
    residuals = res["residuals"]
    it = res["iter"]
    ax[0].plot(residuals, lw=2, color=colors[i])



norm = mpl.colors.Normalize(vmin=omega_ansatz(np.min(alpha_values), N), vmax=omega_ansatz(np.max(alpha_values), N))
sm = plt.cm.ScalarMappable(cmap=cm_r, norm=norm)
cbar = plt.colorbar(sm, ax=ax[0])
cbar.set_label(r"$\omega$")

ax[0].set_yscale("log")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Residual")
ax[0].set_title("Convergence of Residuals")

for i, res in enumerate(results):
    omega = res["omega"]
    it = res["iter"]
    ax[1].scatter(omega, it, color=colors[i], s=25, edgecolor="k", lw=0.25)

norm = mpl.colors.Normalize(vmin=Cs.min(), vmax=Cs.max())
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label("Poiseuille Coefficient $C$")
ax[1].set_xlabel(r"$\omega$")
ax[1].set_ylabel("Iterations to Convergence")
ax[1].set_yscale("log")
ax[1].set_title("Iterations to Convergence")

plt.suptitle("SOR Convergence for Circle Domain")
plt.tight_layout()
plt.subplots_adjust(top=0.82)
parstr = f"Evaluated with: $N={N}$, $\\alpha_0 = {alpha_values.min():.2f}$, $\\alpha_1 = {alpha_values.max():.2f}$, $N_{{\\alpha}} = {len(alpha_values)}$ on 49 Chunks/MPI Ranks"
plt.figtext(0.5, 0.92, parstr, ha='center', va='center', transform=plt.gcf().transFigure, fontsize=12, weight="medium")
plt.savefig("./Images/circle_scan_results.pdf", dpi=450)
plt.show()