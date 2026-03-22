"""
Test WIP Code. Part 2.
"""
import numpy as np
from src import *
from rich import print
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

mask = make_house_mask(1001, pad=1)
mask = np.flip(mask, axis=1)
plt.imshow(mask, cmap='gray', zorder=3)
plt.title("House mask")
plt.show()