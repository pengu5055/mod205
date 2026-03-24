"""
Sweep over the parameter alpha for the omega ansatz:
    \omega(alpha) = 2/(1 + (pi*alpha/N))
"""
import numpy as np
from src import *
from rich import print
from mpi4py import MPI
from time import time
import h5py as h5
import hdf5plugin

N = 404
a0 = 0.1
a1 = N / np.pi
num_points = 3
pad = 1

if __name__ == "__main__":
    ts = time()
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fn = f"./Data/house_scan_a{a0}_g{num_points}_size{size}_N{N}.h5"

    # Set up scan parameters
    PLACEHOLDER = 1.0
    alpha = np.linspace(a0, a1, num_points)
    omega_ansatz = lambda alpha, N: 2 / (1 + (np.pi * alpha / N))

    if rank == 0:
        print(f"Running with {size} ranks.")
        
        mask = make_house_mask(N, pad=pad)
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=PLACEHOLDER, chunks=int(np.sqrt(size)), verbose=True)
        lattice.MAX_ITER = 100000

    for i, a in enumerate(alpha):
        if rank == 0:
            print(f"Running alpha={a:.3f}, omega={omega_ansatz(a, N):.3f} ({i+1}/{len(alpha)})")
            lattice.reset(omega_ansatz(a, N))
            params = lattice.scatter()
        else:
            params = None

        # Instantiate local chunk on each rank
        chunk = SORChunk(comm, params, verbose=True)
        chunk.bar = False
        chunk.run()
        chunk.log.info(f"Took {time() - ts:.2f} seconds for alpha={a:.3f}")

        if rank == 0:
            C = lattice.poiseuille_coeff(chunk.state)
            print(f"Poiseuille coefficient: {C:.4f}")
            print(f"Total Time Taken: {time() - ts:.2f} s")

            try:
                final_state = lattice.state
            except AttributeError:
                print(f"[bold red] WARNING: No final state found on rank 0. Skipping HDF5 write for alpha={a:.2f}.[/bold red]")

            # Append results to HDF5
            with h5.File(fn, "a") as f:
                group_name = f"alpha_{a:.2f}_{i}"
                if group_name in f:
                    del f[group_name]

                group = f.create_group(group_name)
                group.attrs["alpha"] = a
                group.attrs["omega"] = omega_ansatz(a, N)
                group.attrs["poiseuille_coeff"] = C
                group.attrs["iter"] = chunk.iter
                group.create_dataset("state", data=chunk.state,
                                     compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
                group.create_dataset("residuals", data=chunk.residuals,
                                     compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
