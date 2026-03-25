"""
Plot results from house-optimal.py
"""
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from src import *

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

fn = f"./Data/house_optimal_size4_N404.h5"
labels = [
    "Chebyshev Acc. with\nSquare Approx. $\\rho_{{jacobi}}$",
    "$\omega_{\\text{opt}}$ w/o Chebyshev",
    "Chebyshev Acc. with\n$\\rho_{{jacobi}}$ from $\omega_{\\text{opt}}$",
]
residuals = []
iters = []
Cs = []
rhos = []
with h5.File(fn, "r") as f:
    for key in f.keys():
        group = f[key]
        print(f"Loaded group '{key}' with {group.attrs['iter']}")
        residuals.append(group["residuals"][:])
        iters.append(group.attrs["iter"])
        Cs.append(group.attrs["poiseuille_coeff"])
        rhos.append(group.attrs["rho_jacobi"])

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0.0, 0.8))
cs = [0, 2, -2]
cm = cmr.get_sub_cmap("cmr.tropical", 0.0, 0.8)
cm_r = cmr.get_sub_cmap("cmr.tropical_r", 0.2, 1.0)

for i, res in enumerate(residuals):
    ax.plot(res, lw=2, color=colors[cs[i]], label=labels[i]) 

ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual")
ax.set_title("Convergence of Residuals")
ax.legend()

plt.suptitle("SOR Convergence w/ Chebyshev Acc. on 'House' Domain")
plt.tight_layout()
plt.subplots_adjust(top=0.81)
parstr = f"Evaluated with: $N=404$, $\\rho_{{jac}}^{{(num)}} = {sf2tex(rhos[2], 5)}$, $\\rho_{{jac}}^{{(square)}} = {sf2tex(rhos[0], 5)}$ on 4 Chunks/MPI Ranks" 
plt.figtext(0.5, 0.91, parstr, ha='center', va='center', transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.savefig("./Images/house_optimal_comparison.pdf", dpi=450)
plt.show()