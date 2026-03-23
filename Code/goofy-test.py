"""
Feed the SORLattice arbitrary bool masks.
"""
import numpy as np
from src import *
from rich import print
from mpi4py import MPI
from time import time
from PIL import Image

if __name__ == "__main__":
    ts = time()
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running with {size} ranks.")
        img = Image.open("./Sources/mm2.jpg").convert("L")
        N = 600
        img = img.resize((N, N))
        mask = np.array(img) < 170
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        if False:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            mpl.use("tkagg")
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(mask, cmap="gray")
            ax[0].set_title("Domain Mask")
            ax[1].imshow(img, cmap="gray")
            ax[1].set_title("Original Image")
            plt.show()
            quit()

        lattice = SORLattice(comm=comm, domain_mask=mask, tol=1e-10,
                             dx=1/N, rhs_value=-1.0, omega=1.9, chunks=int(np.sqrt(size)), verbose=True)
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
        np.savez("goofy-test.npz", state=lattice.state, mask=mask, residuals=chunk.residuals, poiseuille_coeff=C)
        print(f"Total Time Taken: {time() - ts:.2f} s")
