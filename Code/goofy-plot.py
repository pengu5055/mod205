"""
Plot goofy gag data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

data = np.load("goofy-test.npz")
state = data["state"]
mask = data["mask"]
residuals = data["residuals"]
C = data["poiseuille_coeff"]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

cm = cmr.get_sub_cmap("cmr.chroma", 0.1, 0.75)
cm.set_bad(color="lightgray", alpha=0.0)
state /= np.max(state[mask])
state[state < 0.01] = np.nan

norm = mpl.colors.LogNorm(vmin=0.01, vmax=np.nanmax(state[mask]))
im = ax[0].imshow(state, cmap=cm, norm=norm, zorder=4)
ax[0].imshow(mask, cmap="gray", alpha=1.0, zorder=3)
ax[0].set_title(f"Goofy Gag State (C={C:.4f})")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
ax[0].grid(False)

ax[1].plot(residuals, marker="o", markersize=3)
ax[1].set_yscale("log")
ax[1].set_title("Residuals")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Residual")

plt.tight_layout()
plt.show()
