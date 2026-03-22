"""
Contains fuctions to be used throughout the project.
"""
import numpy as np
from collections import deque
from typing import Iterable
from mpi4py import MPI
from rich import print as print
from rich.progress import Progress
from skimage.measure import find_contours
import h5py as h5
import hdf5plugin

class SORLattice:
    """
    Class for root rank to instantiate the full SOR grid and partition it into chunks for each rank.
    """
    def __init__(self, 
                 domain_mask: np.ndarray,
                 dx: float,
                 rhs_value: float,
                 omega: float,
                 chunks: int,
                 tol: float = 1e-6,
                 verbose: bool = False) -> None:
        """
        Constructor for the full SOR grid on the root rank. Partitions the grid into chunks and creates a topology.

        Parameters
        ----------
        global_dimensions : tuple
            Dimensions of the full grid (x_dim, y_dim).
        chunks : int 
            Number of chunks to partition the grid into along each dimension (e.g. 2 means 2x2=4 total chunks).
        verbose : bool, optional
            Whether to print verbose output during initialization and execution, by default False.
        """
        print(f"[bold green][Rank 0][/bold green] Initializing SORLattice with {chunks}x{chunks} chunks...")
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        assert self.rank == 0, "SORLattice should only be instantiated on the root rank (rank 0)."

        self.dx = dx
        self.tol = tol
        self.omega = omega
        self.domain_mask = domain_mask.astype(bool)
        self.chunks = chunks
        self.verbose = verbose
        self.CLIP_MIN = 1e-3
        
        # Create topology as a 2D grid of ranks
        self.topology = np.arange(self.chunks**2).reshape((self.chunks, self.chunks))
        self.chunks_shape = self.topology.shape

        # Full domain arrays
        self.domain_mask_full = self.domain_mask.astype(bool)
        self.full_shape = self.domain_mask_full.shape
        self.x_g_dim, self.y_g_dim = self.full_shape

        # Assert grid shape
        if self.x_g_dim % self.chunks != 0 or self.y_g_dim % self.chunks != 0:
            raise ValueError(f"Grid dimensions {self.full_shape} must be divisible by number of chunks {self.chunks}.")
        
        self.chunk_x = self.x_g_dim // self.chunks
        self.chunk_y = self.y_g_dim // self.chunks

        # Create RHS Field
        self.RHS = np.where(self.domain_mask_full, rhs_value * dx**2 / 2, 0.0).astype(np.float64)

        # Compute theta arrays for Shortley-Weller stencil
        if self.verbose:
            print(f"[bold green][Rank 0][/bold green] Computing theta arrays for Shortley-Weller stencil...")

        (self.theta_right, self.theta_left, self.theta_up, self.theta_down) = self._compute_theta_arrays(self.dx)

        # Solution field. Zero all since BC's are enforced via masking
        self.state = np.zeros_like(self.domain_mask_full, dtype=np.float64)

    def _compute_theta_arrays(self, dx: float):
        """
        Compute the theta arrays for the Shortley-Weller stencil.
        For simplicity, we assume a uniform grid. Use the domain_mask stencil 
        to determine values
        """
        # Initialize Full theta arrays to 1.0
        rows, cols = self.full_shape
        theta_right = np.ones(self.full_shape, dtype=np.float64)
        theta_left = np.ones(self.full_shape, dtype=np.float64)
        theta_up = np.ones(self.full_shape, dtype=np.float64)
        theta_down = np.ones(self.full_shape, dtype=np.float64)

        # Find vertices in pixel coordinates (i.e. corners of the grid cells)
        contours = find_contours(self.domain_mask.astype(float), level=0.5)

        # Convert pixel coordinates to physical coordinates
        segments = []
        rows = self.domain_mask.shape[0]
        for contour in contours:
            for k in range(len(contour) - 1):
                ax = (rows - 1 - contour[k, 0]) * dx
                ay = contour[k, 1] * dx
                bx = (rows - 1 - contour[k + 1, 0]) * dx
                by = contour[k + 1, 1] * dx

                segments.append(((ax, ay), (bx, by)))

        # Thus physical coords
        x_coords = np.arange(cols) * self.dx
        y_coords = (rows - 1 - np.arange(rows)) * self.dx

        # Mask only boundary-adjacent interior points
        mask = self.domain_mask
        boundary_adjacent = np.zeros_like(mask, dtype=bool)
        boundary_adjacent[1:-1, 1:-1] = (
                mask[1:-1, 1:-1] & (
                ~mask[0:-2, 1:-1] |
                ~mask[2:,   1:-1] |
                ~mask[1:-1, 0:-2] |
                ~mask[1:-1, 2:  ]))
        
        # Directions defined as (ddx, ddy, array, di, dj)
        directions = [
            (self.dx, 0.0, theta_right, 0, 1),   # Right
            (-self.dx, 0.0, theta_left, 0, -1),  # Left
            (0.0, self.dx, theta_up, -1, 0),     # Up
            (0.0, -self.dx, theta_down, 1, 0)    # Down
        ]

        # Now run update loop for each boundary-adjacent interior point
        for i, j in zip(*np.where(boundary_adjacent)):
            # WARNING: Notice indices here
            px = x_coords[j]
            py = y_coords[i]

            for ddx, ddy, theta_array, di, dj in directions:
                ni, nj = i + di, j + dj
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue  # Skip out-of-bounds neighbors
                if mask[ni, nj]:  
                    continue

                # Find closest boundary intersections in this direction
                t_min = None
                for (ax, ay), (bx, by) in segments:
                    t = self._ray_intersect_segment(px, py, ddx, ddy, ax, ay, bx, by)
                    if t is None or t < t_min:
                        t_min = t

                if t_min is not None:
                    theta_array[i, j] = np.clip(t_min / self.dx, self.CLIP_MIN, 1.0)
                    

        return theta_right, theta_left, theta_up, theta_down

class SORChunk:
    """
    A class to instantiate a chunk of the SOR grid.
    """
    def __init__(self, 
                 topology: Iterable,
                 dimensions: tuple[int, int],
                 verbose: bool = False) -> None:
        """
        MPI-based constructor for a chunk of the SOR grid.

        state.shape = (x_dim, y_dim)
        ^ includes:
            - ghost row at top    [0, :]
            - ghost row at bottom [-1, :]
            - ghost column left   [:, 0]
            - ghost column right  [:, -1]

        RHS = q * dx^2/2 for Poisson's equation with source term q. 
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Determine neighbors from topology
        self.topology = topology
        if self.topology.ndim != 1:
            self.chunk_loc = np.where(topology == self.rank)
            self.chunk_loc = (int(self.chunk_loc[0][0]), int(self.chunk_loc[1][0]))
            self.chunks = topology.shape[0]
            self.loc = topology[self.chunk_loc[0], self.chunk_loc[1]]
            self.n_up = self.topology[self.chunk_loc[0] - 1, self.chunk_loc[1]] if self.chunk_loc[0] > 0 else self.topology[-1, self.chunk_loc[1]]
            self.n_down = self.topology[self.chunk_loc[0] + 1, self.chunk_loc[1]] if self.chunk_loc[0] < self.chunks - 1 else self.topology[0, self.chunk_loc[1]]
            self.n_left = self.topology[self.chunk_loc[0], self.chunk_loc[1] - 1] if self.chunk_loc[1] > 0 else self.topology[self.chunk_loc[0], -1]
            self.n_right = self.topology[self.chunk_loc[0], self.chunk_loc[1] + 1] if self.chunk_loc[1] < self.chunks - 1 else self.topology[self.chunk_loc[0], 0]
            self.n_neighbors = [self.n_up, self.n_down, self.n_left, self.n_right]
        else:
            self.chunk_loc = (0, 0)
            self.chunks = topology.shape[0]
            self.n_up = topology[0]
            self.n_down = topology[-1]
            self.n_left = topology[0]
            self.n_right = topology[-1]
            self.n_neighbors = [self.n_up, self.n_down, self.n_left, self.n_right]

        self.x_dim, self.y_dim = dimensions
        if self.x_dim % 2 != 0 or self.y_dim % 2 != 0:
            raise ValueError("Dimensions must be even/divisible by 2.")
        self.x_dim += 2  # Add ghost rows
        self.y_dim += 2  # Add ghost columns

        # Various internal variables
        self.zerodiv = 1e-14
        self.RHS = 0.0
        self.MAX_ITER = 10000

        self.init_state = None
        self.state = None
        self._randomize_state()

        print(f"[bold green][Rank {self.rank}][/bold green] Ready! My location is " +
              f"{self.chunk_loc} with neighbors {self.n_up}, {self.n_down}, {self.n_left}, {self.n_right}.")
        
    def _randomize_state(self) -> None:
        self.state = np.random.rand(self.x_dim, self.y_dim)
        self.init_state = self.state.copy()

    def _broadcast_borders(self):
        """
        Broadcast the borders of the lattice to neighboring processes.
        Due to Checkerboard update we send all borders two times per step.

        Even ranks send first. Odd ranks receive first.
        First horizontal, then vertical communication.
        """
        y_loc, x_loc = self.chunk_loc
        if self.verbose:
            print(f"[bold green][Rank {self.rank}][/bold green] Broadcasting Borders...")

        # Phase 1: horizontal communication
        if x_loc % 2 == 0:
            if self.n_right != self.rank:
                sendbuf = self.state[1:-1, -2].copy()
                self.comm.Send(sendbuf, dest=self.n_right, tag=10)
            if self.n_left != self.rank:
                recvbuf = np.empty_like(self.state[1:-1, 0])
                self.comm.Recv(recvbuf, source=self.n_left, tag=10)
                self.state[1:-1, 0] = recvbuf
    
            if self.n_left != self.rank:
                sendbuf = self.state[1:-1, 1].copy()
                self.comm.Send(sendbuf, dest=self.n_left, tag=11)
            if self.n_right != self.rank:
                recvbuf = np.empty_like(self.state[1:-1, -1])
                self.comm.Recv(recvbuf, source=self.n_right, tag=11)
                self.state[1:-1, -1] = recvbuf
        else:
            if self.n_left != self.rank:
                recvbuf = np.empty_like(self.state[1:-1, 0])
                self.comm.Recv(recvbuf, source=self.n_left, tag=10)
                self.state[1:-1, 0] = recvbuf
            if self.n_right != self.rank:
                sendbuf = self.state[1:-1, -2].copy()
                self.comm.Send(sendbuf, dest=self.n_right, tag=10)
    
            if self.n_right != self.rank:
                recvbuf = np.empty_like(self.state[1:-1, -1])
                self.comm.Recv(recvbuf, source=self.n_right, tag=11)
                self.state[1:-1, -1] = recvbuf
            if self.n_left != self.rank:
                sendbuf = self.state[1:-1, 1].copy()
                self.comm.Send(sendbuf, dest=self.n_left, tag=11)

        if self.verbose:
                print(f"[bold green][Rank {self.rank}][/bold green] Waiting at barrier...")
        self.comm.Barrier()
    
        # Phase 2: vertical communication
        if y_loc % 2 == 0:
            if self.n_down != self.rank:
                sendbuf = self.state[-2, 1:-1].copy()
                self.comm.Send(sendbuf, dest=self.n_down, tag=20)
            if self.n_up != self.rank:
                recvbuf = np.empty_like(self.state[0, 1:-1])
                self.comm.Recv(recvbuf, source=self.n_up, tag=20)
                self.state[0, 1:-1] = recvbuf
    
            if self.n_up != self.rank:
                sendbuf = self.state[1, 1:-1].copy()
                self.comm.Send(sendbuf, dest=self.n_up, tag=21)
            if self.n_down != self.rank:
                recvbuf = np.empty_like(self.state[-1, 1:-1])
                self.comm.Recv(recvbuf, source=self.n_down, tag=21)
                self.state[-1, 1:-1] = recvbuf
        else:
            if self.n_up != self.rank:
                recvbuf = np.empty_like(self.state[1, 1:-1])
                self.comm.Recv(recvbuf, source=self.n_up, tag=20)
                self.state[1, 1:-1] = recvbuf
            if self.n_down != self.rank:
                sendbuf = self.state[-2, 1:-1].copy()
                self.comm.Send(sendbuf, dest=self.n_down, tag=20)
    
            if self.n_down != self.rank:
                recvbuf = np.empty_like(self.state[-1, 1:-1])
                self.comm.Recv(recvbuf, source=self.n_down, tag=21)
                self.state[-1, 1:-1] = recvbuf
            if self.n_up != self.rank:
                sendbuf = self.state[1, 1:-1].copy()
                self.comm.Send(sendbuf, dest=self.n_up, tag=21)
    
            if self.verbose:
                print(f"[bold green][Rank {self.rank}][/bold green] Waiting at barrier...")
        self.comm.Barrier()

    def _sor_step(self):
        """
        Perform one full SOR sweep: Red (parity 0) then Black (parity 1) half-steps with
        ghost border broadcasts in between.
        """
        self._broadcast_borders()
        self._sor_update(parity=0)

        self._broadcast_borders()
        self._sor_update(parity=1)

    def _sor_update(self, parity: int):
        """
        Shortly-Weller adapted SOR update for a given parity (0 for red, 1 for black).

        self.theta_[left,right,up,down] are arrays of shape (x_dim, y_dim) that store the fractional 
        distances to the boundary in each direction:
            theta = 1.0 -> fully interior, 
            0 < theta < 1 -> for points whose neighbor in that direction hits a boundary,
            theta = 0.0 -> points outside the domain.
        
        The Shortley-Weller stencil for (\nabla^2 u = RHS) scaled so the sum of the coefficients is 1 is:
            u_gs = (u_R/theta_R(theta_R + theta_L) + u_L/theta_L(theta_R + theta_L)
                   + u_U/theta_U(theta_U + theta_D) + u_D/theta_D(theta_U + theta_D) - RHS)
        where u_R, u_L, u_U, u_D are short-hand for [left, right, up, down] neighbor values and theta_ are 
        as defined above. 
        """
        u = self.state

        # Shift neighbor values (rolling since ghost borders prevent out-of-bounds)
        u_R = np.roll(u, shift=-1, axis=1)
        u_L = np.roll(u, shift=1, axis=1)
        u_U = np.roll(u, shift=-1, axis=0)
        u_D = np.roll(u, shift=1, axis=0)

        t_R, t_L = self.theta_right, self.theta_left
        t_U, t_D = self.theta_up, self.theta_down

        # Compute the Shortley-Weller wights
        w_R = 1.0 / (t_R * (t_R + t_L) + self.zerodiv)
        w_L = 1.0 / (t_L * (t_R + t_L) + self.zerodiv)
        w_U = 1.0 / (t_U * (t_U + t_D) + self.zerodiv)
        w_D = 1.0 / (t_D * (t_U + t_D) + self.zerodiv)

        w_sum = w_R + w_L + w_U + w_D

        u_gs = (w_R * u_R +
                w_L * u_L +
                w_U * u_U +
                w_D * u_D - self.RHS) / w_sum
        
        u_new = (1 - self.omega) * u + self.omega * u_gs

        # Now only update the points of the given parity
        checkerboard_mask = np.indices(u.shape).sum(axis=0) % 2 == parity

        # Bitwise AND with domain mask to update only interior points (not ghosts, not exterior)
        update_mask = checkerboard_mask & self.domain_mask

        self.state[update_mask] = u_new[update_mask]

    def _global_residual(self):
        """
        Compute the global L2 residual of Poisson's equation across all ranks.
        Use the same Shortley-Weller weights as the update. Only use interior points.
        """
        u = self.state

        # Shift neighbor values (rolling since ghost borders prevent out-of-bounds)
        u_R = np.roll(u, shift=-1, axis=1)
        u_L = np.roll(u, shift=1, axis=1)
        u_U = np.roll(u, shift=-1, axis=0)
        u_D = np.roll(u, shift=1, axis=0)

        t_R, t_L = self.theta_right, self.theta_left
        t_U, t_D = self.theta_up, self.theta_down

        w_R = 1.0 / (t_R * (t_R + t_L) + self.zerodiv)
        w_L = 1.0 / (t_L * (t_R + t_L) + self.zerodiv)
        w_U = 1.0 / (t_U * (t_U + t_D) + self.zerodiv)
        w_D = 1.0 / (t_D * (t_U + t_D) + self.zerodiv)

        w_sum = w_R + w_L + w_U + w_D

        laplacian = (w_R * u_R + 
                     w_L * u_L +
                     w_U * u_U +
                     w_D * u_D) / w_sum
        
        residual_field = (laplacian - u - self.RHS)
        local_l2 = np.sum(residual_field[self.domain_mask] ** 2)

        # comm.allreduce to get global L2 residual
        global_l2 = self.comm.allreduce(local_l2, op=MPI.SUM)

        return np.sqrt(global_l2)

    def _gather_chunks(self):
        """
        Gather all chunks to the root process.
        """
        # Dirty Fix: Subtract 2 from x_dim and y_dim to remove ghost rows and columns
        self.x_dim -= 2
        self.y_dim -= 2

        if self.verbose:
            print(f"[bold green][Rank {self.rank}][/bold green] Final State Sum: {np.sum(self.state)}")
            print(f"[bold green][Rank {self.rank}][/bold green] Local state has NaN: {np.isnan(self.state).any()}")
        if self.rank == 0:
            print(f"[bold green][Rank {self.rank}][/bold green] {self.x_dim} x {self.y_dim} grid with {self.size} chunks.")
            self.chunks_s_init = np.zeros((self.size, self.x_dim, self.y_dim), dtype=self.state.dtype)
            self.chunks_s_final = np.zeros((self.size, self.x_dim, self.y_dim), dtype=self.state.dtype)
            self.comm.Gather(self.init_state, self.chunks_s_init, root=0)
            self.comm.Gather(self.state, self.chunks_s_final, root=0)

            self.full_s_final = np.zeros((self.chunks * self.x_dim, self.chunks * self.y_dim), dtype=self.state.dtype)
            self.full_s_init = np.zeros((self.chunks * self.x_dim, self.chunks * self.y_dim), dtype=self.state.dtype)

            # Reconstruct the full grid
            for rank in range(self.size):
                row = rank // self.chunks
                col = rank % self.chunks
                self.full_s_final[
                    row * self.x_dim : (row + 1) * self.x_dim,
                    col * self.y_dim : (col + 1) * self.y_dim
                ] = self.chunks_s_final[rank]
                self.full_s_init[
                    row * self.x_dim : (row + 1) * self.x_dim,
                    col * self.y_dim : (col + 1) * self.y_dim
                ] = self.chunks_s_init[rank]
        
            return self.full_s_init, self.full_s_final
        else:
            self.comm.Gather(self.init_state, None, root=0)
            self.comm.Gather(self.state, None, root=0)

    def run(self):
        """
        Run the SOR algorithm.
        """
        # Wait for all ranks to initialize
        self.comm.Barrier()

        if self.rank == 0 and self.verbose:
            with Progress() as p:
                task = p.add_task(f"[bold green][All Ranks][/bold green]", total=self.MAX_ITER)
                while self.iter < self.MAX_ITER:
                    self._sor_step()
                    residual = self._global_residual()
                    self.residuals.append(residual)
                    self.iter += 1
                    p.update(task, advance=1)

                    if residual < self.tol:
                        print(f"[bold green][Rank 0][/bold green] Converged at iteration {self.iter}, residual {residual:.2e}")
                        break
        else:
            while self.iter < self.MAX_ITER:
                self._sor_step()
                residual = self._global_residual()
                self.residuals.append(residual)
                self.iter += 1
                
                if residual < self.tol:
                    break
        
        # Wait for all ranks to finish
        self.comm.Barrier()
        
        # Truncate state to original dimensions (remove ghost rows and columns)
        self.state = np.ascontiguousarray(self.state[1:-1, 1:-1])
        self.init_state = np.ascontiguousarray(self.init_state[1:-1, 1:-1])

        if self.verbose:
            print(f"[bold green][Rank {self.rank}][/bold green] Simulation Complete. " +  
                  f"Final Energy: {self.energy}, Final Temperature: {self.temperature}, Total Iterations: {self.iter}")


        return self.init_state, self.state, self._energy_list
