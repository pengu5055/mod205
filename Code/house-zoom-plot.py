"""
Plot results house-scan zoomed in on the optimal alpha area.
"""
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from src import *

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

N = 404
size = 4
n_points = 50
a_start = 0.1
omega_ansatz = lambda alpha, N: 2 / (1 + (np.pi * alpha / N))
fn = f"./Data/house_scan_a{a_start}_g{n_points}_size{size}_N{N}.h5"

results = load_scan(fn)
print(f"Loaded {len(results)} results from {fn}")
alpha_values = np.array([res["alpha"] for res in results])
omega_values = np.array([res["omega"] for res in results])
iters = np.array([res["iter"] for res in results])
Cs = np.array([res["C"] for res in results])

# Find the optimal alpha and corresponding omega
optimal_idx = np.argmin(iters)
optimal_alpha = alpha_values[optimal_idx]
optimal_omega = omega_values[optimal_idx]
print(f"Optimal alpha: {optimal_alpha:.8f}, Optimal omega: {optimal_omega:.8f}, Iterations: {iters[optimal_idx]}, C: {Cs[optimal_idx]:.8f}")


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

sc = ax[1].scatter(omega_values, iters, c=Cs, cmap=cm, s=25, edgecolor="k", lw=0.1, norm=mpl.colors.LogNorm(vmin=Cs.min(), vmax=Cs.max()))
cbar = plt.colorbar(sc, ax=ax[1])
cbar.set_label("Poiseuille Coefficient $C$")

ax[1].set_xlabel(r"$\omega$")
ax[1].set_ylabel("Iterations to Convergence")
ax[1].set_yscale("log")
ax[1].set_title("Iterations to Convergence")

plt.suptitle("Zoomed-In SOR Convergence for 'House' Domain")
plt.tight_layout()
plt.subplots_adjust(top=0.82)
parstr = f"Evaluated with: $N={N}$, $\\alpha_0 = {alpha_values.min():.2f}$, $\\alpha_1 = {alpha_values.max():.2f}$, $N_{{\\alpha}} = {len(alpha_values)}$ on {size} Chunks/MPI Ranks"
plt.figtext(0.5, 0.92, parstr, ha='center', va='center', transform=plt.gcf().transFigure, fontsize=12, weight="medium")
plt.savefig("./Images/house_scan_results_zoom.pdf", dpi=450)
plt.show()