"""
Test WIP Code. Part 2.
"""
import numpy as np
from src import *
from rich import print
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from mpi4py import MPI
from time import time

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

if __name__ == "__main__":
    ts = time()

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running with {size} ranks.")

    # Problem parameters
    R    = 1.0
    H    = 2 * R
    flux = 1.0
    Nr   = 100
    Nz   = 200
    dr   = R / (Nr - 1)
    dz   = H / (Nz - 1)
    pad  = 1

    bcs = {
        "top":    "neumann",
        "bottom": "neumann",
        "inner":  "neumann",
        "outer":  "dirichlet",
    }

    if rank == 0:
        mask = make_cylinder_mask(Nr, Nz, bcs, pad=pad)
        print(f"top boundary row (row 1) mask: {mask[1, :].sum()}")
        print(f"top boundary row (row 1) mask neumann case: {mask[1, :].sum()}")
        lattice = CylindricalSORLattice(
            comm=comm,
            domain_mask=mask,
            dr=dr,
            dz=dz,
            R=R,
            H=H,
            flux=flux,
            omega=1.9,
            chunks=int(np.sqrt(size)),
            tol=1e-6,
            bcs=bcs,
            verbose=True,
        )
        lattice.MAX_ITER = 100000
        params = lattice.scatter()
    else:
        params = None

    chunk = CylindricalSORChunk(comm, params, verbose=True)
    chunk.run()

    if rank == 0:
        T = lattice.gather(chunk.state)
        print(f"Total Time Taken: {time() - ts:.2f} s")

        # Strip mask boundary ring
        T_plot = T[1:-1, 1:-1]
        r_plot = lattice.r_coords[1:-1, 1:-1]
        z_plot = lattice.z_coords[1:-1, 1:-1]

        # Plotting
        fig, ax = plt.subplots()
        norm = mpl.colors.Normalize(vmin=T_plot.min()+1e-6, vmax=T_plot.max())
        im = ax.imshow(T_plot.T, origin="lower", extent=[r_plot.min(), r_plot.max(), z_plot.min(), z_plot.max()], aspect="auto", cmap=cmr.ember, norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Temperature")
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        plt.title("Steady-State Temperature Distribution in Cylinder")
        plt.tight_layout()
        plt.savefig("./Images/cylinder_temperature_test2.png", dpi=450)