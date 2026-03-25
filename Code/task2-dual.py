"""
Plot steady state solutions for both task 2 scenarios.
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

R    = 1.0
H    = 2 * R
flux = 1.0
Nr   = 100 * 3
Nz   = 200 * 3
dr   = R / (Nr - 1)
dz   = H / (Nz - 1)

cases = {
    "Dirichlet Top": {
        "top": "dirichlet", "bottom": "neumann",
        "inner": "neumann",
        "outer": {"bottom_half": "dirichlet", "top_half": "dirichlet"}
    },
    "Neumann Top": {
        "top": "dirichlet", "bottom": "neumann",
        "inner": "neumann",
        "outer": {"bottom_half": "dirichlet", "top_half": "neumann"}
    },
}

end_state_solutions = {}
coord_grids = {}

if __name__ == "__main__":
    ts = time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for name, bcs in cases.items():
        if rank == 0:
            print(f"Running case: {name}")
            mask = make_cylinder_mask(Nr, Nz, bcs)
            lattice = CylindricalSORLattice(
                comm=comm, domain_mask=mask,
                dr=dr, dz=dz, R=R, H=H, flux=flux,
                omega=1.9, chunks=int(np.sqrt(size)),
                tol=1e-6, bcs=bcs, verbose=True,
            )
            lattice.MAX_ITER = 100000
            params = lattice.scatter()
        else:
            params = None

        chunk = CylindricalSORChunk(comm, params, verbose=True)
        chunk.run()

        if rank == 0:
            T = lattice.gather(chunk.state)
            end_state_solutions[name] = T
            coord_grids[name] = (lattice.r_coords, lattice.z_coords)
            print(f"Case {name} done. T max={T.max():.4f}")

    if rank == 0:
        print(f"Total Time Taken: {time() - ts:.2f} s")

        fig, ax = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
        cm = cmr.get_sub_cmap("cmr.chroma", 0.0, 0.75)

        for i, (name, T) in enumerate(end_state_solutions.items()):
            r, z = coord_grids[name]
            T_plot = T[1:-1, 1:-1] if T.shape != (Nz, Nr) else T
            r_plot = r[1:-1, 1:-1] if r.shape != (Nz, Nr) else r
            z_plot = z[1:-1, 1:-1] if z.shape != (Nz, Nr) else z

            norm = mpl.colors.Normalize(vmin=T_plot.min(), vmax=T_plot.max())
            im = ax[i].imshow(T_plot, origin="upper", zorder=3,
                              extent=[r_plot.min(), r_plot.max(), z_plot.min(), z_plot.max()], aspect="auto", cmap=cm, norm=norm)
            fig.colorbar(im, ax=ax[i], label="Normalized Temperature")
            ax[i].set_title(name)
            ax[i].set_xlabel("r")
            ax[0].set_ylabel("z")
            ax[i].grid(False)

        plt.suptitle("Steady-State Temperature Distribution in Cylinder")
        plt.tight_layout()
        plt.subplots_adjust(top=0.83)
        parstr = f"$N_r={Nr}$, $N_z={Nz}$, $j={flux}$, $R={R}$, $H={H}$ on {size} Chunks/MPI Ranks"
        plt.figtext(0.5, 0.92, parstr, ha='center', va="center", fontsize=12, weight="medium")
        plt.savefig("./Images/cylinder_solutions.pdf", dpi=450)
        plt.show()
