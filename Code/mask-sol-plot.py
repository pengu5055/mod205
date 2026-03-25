"""
Plot the three masks and three steady-state solutions.
"""
import numpy as np
from src import *
from rich import print
from mpi4py import MPI
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

def make_square_mask(N, pad=1):
    mask = np.zeros((N, N), dtype=bool)
    mask[pad:N-pad, pad:N-pad] = True
    return mask

N = 404
masks = {
    "Square": make_square_mask(N, pad=1),
    "Circle": make_circle_mask(N, pad=1),
    "House": make_house_mask(N, pad=1)
}
end_state_solutions = {}
Cs = {}
if __name__ == "__main__":
    ts = time()
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for name, mask in masks.items():
        if rank == 0:
            print(f"Running {name} mask with {size} ranks.")
            lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=1.9, chebyshev=True, rho_jacobi=np.cos(np.pi/N),
                                 chunks=int(np.sqrt(size)), verbose=True)
            lattice.MAX_ITER = 100000
            params = lattice.scatter()
        else:
            params = None

        # Instantiate local chunk on each rank
        chunk = SORChunk(comm, params, verbose=True)
        chunk.bar = False
        chunk.run()

        if rank == 0:
            C = lattice.poiseuille_coeff(chunk.state)
            print(f"{name} Poiseuille coefficient: {C:.4f}")
            end_state_solutions[name] = lattice.state.copy()
            Cs[name] = C

    if rank == 0:
        print(f"Total Time Taken: {time() - ts:.2f} s")
        
        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
        cm = cmr.get_sub_cmap("cmr.chroma", 0.0, 0.75)

        for i, (name, mask) in enumerate(masks.items()):
            # if name == "Square":
            #     mask[0:2, :] = False
            axes[0, i].imshow(mask, cmap="gray")
            axes[0, i].set_title(f"{name} Mask")
            axes[0, i].axis("off")
            axes[0, i].grid(False)

            im = axes[1, i].imshow(end_state_solutions[name], cmap=cm, zorder=3)
            axes[1, i].set_title(f"{name} Solution ($C = {Cs[name]:.4f}$)")
            axes[1, i].axis("off")
            axes[1, i].grid(False)

        plt.suptitle("Steady-State Solutions for Different Masks")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05, hspace=0.125)
        parstr = f"Evaluated with Chebyshev Acc., $N={N}$, $\\rho_{{jac}} = \\cos(\\pi/N)$ on {size} Chunks/MPI Ranks"
        plt.figtext(0.5, 0.01, parstr, ha='center', va='bottom', transform=plt.gcf().transFigure, fontsize=12, weight="medium")
        plt.savefig("./Images/mask_solutions2.pdf", dpi=450)
        plt.close()
        # plt.show()
