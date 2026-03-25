"""
Task 2a: Cylinderical SOR with Upper Half Dirichlet 
"""
import numpy as np
from src import *
from rich import print
from mpi4py import MPI
from time import time

# Problem parameters
R    = 1.0
H    = 2 * R
flux = 1.0
Nr   = 101
Nz   = 200
dr   = R / (Nr - 1)
dz   = H / (Nz - 1)
a0   = 0.1
a1   = 4.0
num_points = 3

bcs = {
        "top":    "dirichlet",
        "bottom": "neumann",
        "inner":  "neumann",
        "outer": {
            "bottom_half": "dirichlet",
            "top_half":    "neumann",
        }
      }

if __name__ == "__main__":
    ts = time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fn = f"./Data/task2_a{a0}_g{num_points}_size{size}_Nr{Nr}_th-bc{bcs['outer']['top_half']}.h5"

    # Set up scan parameters
    PLACEHOLDER = 1.0
    alpha = np.linspace(a0, a1, num_points)
    omega_ansatz = lambda alpha, N: 2 / (1 + (np.pi * alpha / N))

    if rank == 0:
        print(f"Running with {size} ranks.")
        mask = make_cylinder_mask(Nr, Nz, bcs)
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

    for i, a in enumerate(alpha):
        if rank == 0:
            print(f"Running alpha={a:.3f}, omega={omega_ansatz(a, Nr):.3f} ({i+1}/{len(alpha)})")
            lattice.reset(omega_ansatz(a, Nr))
            params = lattice.scatter()
        else:
            params = None

        # Instantiate local chunk on each rank
        chunk = CylindricalSORChunk(comm, params, verbose=True)
        chunk.run()
        chunk.log.info(f"Took {time() - ts:.2f} seconds for alpha={a:.3f}")

        if rank == 0:
            try:
                T = lattice.gather(chunk.state)
            except AttributeError:
                print(f"[bold red] WARNING: No final state found on rank 0. Skipping HDF5 write for alpha={a:.2f}.[/bold red]")
                continue

            # Append results to HDF5
            with h5.File(fn, "a") as f:
                group_name = f"alpha_{a:.2f}_{i}"
                if group_name in f:
                    del f[group_name]

                group = f.create_group(group_name)
                group.attrs["alpha"] = a
                group.attrs["omega"] = omega_ansatz(a, Nr)
                group.attrs["iter"] = chunk.iter
                group.attrs["time"] = time() - ts
                group.create_dataset("state", data=T,
                                     compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
                group.create_dataset("residuals", data=chunk.residuals,
                                     compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))