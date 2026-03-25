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

fn = "./Data/square_scan_a0.0_g50_size4_N404.h5"

results = load_scan(fn)

state = results[0]['state']

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(state, origin='lower', cmap=cmr.ember)
ax.set_title(f"Final state for alpha={results[0]['alpha']:.2f}, omega={results[0]['omega']:.2f}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.colorbar(im, cmap=cmr.ember, ax=ax, label="Flow Rate")
plt.tight_layout()
plt.show()