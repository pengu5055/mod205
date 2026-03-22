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
        half_interior = 99 * 10
        pad = 1
        N = 2 * half_interior + 2 * pad
        mask = make_circle_mask(N, pad=pad)
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=1/N, rhs_value=-1.0, omega=1.9, chunks=2, verbose=True)
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
        print(f"Poiseuille coefficient: {C:.4f}")
