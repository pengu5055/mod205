"""
Test WIP Code.
"""
import numpy as np
from src import *
from rich import print
from mpi4py import MPI

if __name__ == "__main__":
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        N = 100
        mask = np.ones((N, N), dtype=bool)
        mask[0, :]  = False
        mask[-1, :] = False
        mask[:, 0]  = False
        mask[:, -1] = False 
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=1.5, chunks=1, verbose=True)
        params = lattice.scatter()
    else:
        params = None

    # Instantiate local chunk on each rank
    chunk = SORChunk(comm, params)
    chunk.run()

    if rank == 0:
        C = lattice.poiseuille_coeff(chunk.state)
        print(f"Poiseuille coefficient: {C:.4f}")
