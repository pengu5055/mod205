"""
Code to run the solver.
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
        mask = np.ones((100, 100), dtype=bool)  
        lattice = SORLattice(comm=comm, domain_mask=mask, dx=0.1, rhs_value=-1.0, omega=1.5, chunks=1, verbose=True)
        params = lattice.scatter()
    else:
        params = None

    # Instantiate local chunk on each rank
    chunk = SORChunk(comm, params)
    chunk.run()

    if rank == 0:
        lattice.gather(chunk.state)
        C = lattice.poiseuille_coeff()
        print(f"Poiseuille coefficient: {C:.4f}")
