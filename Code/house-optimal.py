"""
Compare convergence for the optimal omega we found vs. optimal 
omega fed as rho_jacobi for Chebyshev acceleration.
"""
import numpy as np
from src import *
from rich import print
from mpi4py import MPI
from time import time

OMEGA_OPT = 1.97641325
half_interior = 99
pad = 1
N = 404

if __name__ == "__main__":
    ts = time()
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fn = f"./Data/house_optimal_size{size}_N{N}.h5"

    if rank == 0:
        print(f"Running with {size} ranks.")
        mask = make_house_mask(N, pad=pad)
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=OMEGA_OPT, chebyshev=False,
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

        # Append results to HDF5
        with h5.File(fn, "a") as f:
            group_name = f"no_chebyshev_{OMEGA_OPT:.3f}"
            if group_name in f:
                del f[group_name]

            group = f.create_group(group_name)
            group.attrs["omega"] = OMEGA_OPT
            group.attrs["poiseuille_coeff"] = C
            group.attrs["iter"] = chunk.iter
            group.attrs["time"] = time() - ts
            group.attrs["rho_jacobi"] = 0.0
            group.create_dataset("state", data=lattice.state,
                                    compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
            group.create_dataset("residuals", data=chunk.residuals,
                                    compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
            
    # This feels dirty to not do as a loop but eh..
    if rank == 0:
        print(f"Running with rho_jacobi = {np.cos(np.pi/N):.5f} on {size} ranks.")
        half_interior = 99
        pad = 1
        N = 2 * half_interior + 2 * pad
        mask = make_house_mask(N, pad=pad)
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=OMEGA_OPT, chebyshev=True, rho_jacobi=np.cos(np.pi/N),
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

        # Append results to HDF5
        with h5.File(fn, "a") as f:
            group_name = f"generic_chebyshev_{OMEGA_OPT:.3f}"
            if group_name in f:
                del f[group_name]

            group = f.create_group(group_name)
            group.attrs["omega"] = 1.0
            group.attrs["poiseuille_coeff"] = C
            group.attrs["iter"] = chunk.iter
            group.attrs["time"] = time() - ts
            group.attrs["rho_jacobi"] = np.cos(np.pi/N)
            group.create_dataset("state", data=lattice.state,
                                    compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
            group.create_dataset("residuals", data=chunk.residuals,
                                    compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
            

    # Blegh disgusting code reuse
    if rank == 0:
        print(f"Running with rho_jacobi = {omega_2_jacobi(OMEGA_OPT):.5f} on {size} ranks.")
        half_interior = 99
        pad = 1
        N = 2 * half_interior + 2 * pad
        mask = make_house_mask(N, pad=pad)
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=OMEGA_OPT, chebyshev=True, rho_jacobi=omega_2_jacobi(OMEGA_OPT),
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

        # Append results to HDF5
        with h5.File(fn, "a") as f:
            group_name = f"optimal_chebyshev_{OMEGA_OPT:.3f}"
            if group_name in f:
                del f[group_name]

            group = f.create_group(group_name)
            group.attrs["omega"] = 1.0
            group.attrs["poiseuille_coeff"] = C
            group.attrs["iter"] = chunk.iter
            group.attrs["time"] = time() - ts
            group.attrs["rho_jacobi"] = omega_2_jacobi(OMEGA_OPT)
            group.create_dataset("state", data=lattice.state,
                                    compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))
            group.create_dataset("residuals", data=chunk.residuals,
                                    compression=hdf5plugin.Blosc2("zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE))