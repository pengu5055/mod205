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

def make_house_mask(N: int, pad: int = 1) -> np.ndarray:
    """
    Build boolean domain mask for the house+cross shape on an NxN grid.
    (N - 2 * pad) must be divisible by 3.

        3x3 sub-square layout:
        [0=TL-out, 1=TR-out, 2=out]
        [3=in,     4=in,     5=in ]
        [6=out,    7=in,     8=out]
    
        Peak of roof at (row=0, col=2s) — shared corner of chunks 0 and 1.
        
        
    
    Parameters
    ----------
    N : int
        The size of the grid including the padding.
    pad : int
        The width of the padding around the grid. 
        The actual interior domain will be (N - 2*pad) x (N - 2*pad).
    """
    inner = N - 2 * pad
    assert inner % 3 == 0, f"N - 2*pad = {inner} must be divisible by 3."
    si = inner // 3

    mask_inner = np.zeros((inner, inner), dtype=bool)
    rows, cols = np.indices((inner, inner))

    # Middle row: chunks 3, 4, 5 — fully interior
    mask_inner[si:2*si, :] = True

    # Bottom center: chunk 7 — fully interior
    mask_inner[2*si:3*si, si:2*si] = True

    # Chunk 0 diagonal: (row=0, col=2s) -> (row=s, col=s)   top-left cut out
    r0 = rows[0:si, si:2*si]
    c0 = cols[0:si, si:2*si]
    mask_inner[0:si, si:2*si] = (r0 + c0) >= 2*si

    # Chunk 1 diagonal: (row=0, col=2s) -> (row=s, col=3s)  top-right cut out
    r1 = rows[0:si, 2*si:3*si]
    c1 = cols[0:si, 2*si:3*si]
    mask_inner[0:si, 2*si:3*si] = r1 >= (c1 - 2*si)

    # Embed in padded array
    mask = np.zeros((N, N), dtype=bool)
    mask[pad:-pad, pad:-pad] = mask_inner

    # I flipped the indices arrr
    return np.flip(mask, axis=1)

def make_circle_mask(N: int, pad: int = 1) -> np.ndarray:
    total = N + 2 * pad
    mask = np.zeros((total, total), dtype=bool)
    cx, cy = total / 2, total / 2
    R = N / 2
    rows, cols = np.indices((total, total))
    mask = (rows - cx)**2 + (cols - cy)**2 < R**2

    return mask

class SORLattice:
    """
    Class for root rank to instantiate the full SOR grid and partition it into chunks for each rank.
    """
    def __init__(self, 
                 comm: MPI.Comm,
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
        
        """
        print(f"[bold green][Rank 0][/bold green] Initializing SORLattice with {chunks}x{chunks} chunks...")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        assert self.rank == 0, "SORLattice should only be instantiated on the root rank (rank 0)."

        # Never liked this tab style format but really helps readability heh..
        self.dx          = dx
        self.tol         = tol
        self.omega       = omega
        self.domain_mask = domain_mask.astype(bool)
        self.chunks      = chunks
        self.verbose     = verbose
        self.CLIP_MIN    = 1e-3
        
        # Create topology as a 2D grid of ranks
        self.topology = np.arange(self.chunks**2).reshape((self.chunks, self.chunks))
        self.chunks_shape = self.topology.shape
        self.n_chunks = self.topology.size

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
        rows, cols  = self.full_shape
        theta_right = np.ones(self.full_shape, dtype=np.float64)
        theta_left  = np.ones(self.full_shape, dtype=np.float64)
        theta_up    = np.ones(self.full_shape, dtype=np.float64)
        theta_down  = np.ones(self.full_shape, dtype=np.float64)

        # Find vertices in pixel coordinates (i.e. corners of the grid cells)
        contours = find_contours(self.domain_mask.astype(float), level=0.5)

        # Convert pixel coordinates to physical coordinates
        segments = []
        rows = self.domain_mask.shape[0]
        for contour in contours:
            for k in range(len(contour) - 1):
                ax = contour[k, 1] * dx
                ay = (rows - 1 - contour[k, 0]) * dx
                bx = contour[k + 1, 1] * dx
                by = (rows - 1 - contour[k + 1, 0]) * dx

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
                    continue  # Neighbor is interior, theta already 1.0

                # Find closest boundary intersections in this direction
                t_min = None
                for (ax, ay), (bx, by) in segments:
                    t = self._ray_intersect_segment(px, py, ddx, ddy, ax, ay, bx, by)
                    if t is None:
                        if t_min is None or t < t_min:
                            t_min = t

                if t_min is not None:
                    theta_array[i, j] = np.clip(t_min / self.dx, self.CLIP_MIN, 1.0)


        return theta_right, theta_left, theta_up, theta_down
    
    def _ray_intersect_segment(self, px, py, ddx, ddy, ax, ay, bx, by):
        """
        Compute the intersection of a ray from (px, py) in direction (ddx, ddy) with a line segment from (ax, ay) to (bx, by).
        Returns the distance t along the ray to the intersection point, or None if no intersection.

            Ray = P + t *(ddx, ddy)
            Segment: A + s * (B - A) for s in [0, 1]
            Need to solve for t and s such that:
                P + t * (ddx, ddy) = A + s * (B - A)
        """
        # Line segment vector
        sx = bx - ax
        sy = by - ay

        # Solve for intersection using Cramer's rule
        det = ddx * (-sy) - ddy * (-sx)
        if abs(det) < 1e-14:
            return None  # Parallel lines

        t = ((ax - px) * (-sy) - (ay - py) * (-sx)) / det
        u = ((ax - px) * ddy - (ay - py) * ddx) / det

        if t >= 0 and 0 <= u <= 1:
            return t
        else:
            return None

    def _get_chunk_slice(self, rank: int):
        """
        Get the idx (row_start, row_end, col_start, col_end) 
        of interor data for a given chunk rank.
        """
        loc = np.where(self.topology == rank)
        row = int(loc[0][0])
        col = int(loc[1][0])
        r_start = row * self.chunk_x
        r_end = r_start + self.chunk_x
        c_start = col * self.chunk_y
        c_end = c_start + self.chunk_y

        return r_start, r_end, c_start, c_end

    def _add_ghost_borders(self, arr, r_start, r_end, c_start, c_end):
        """
        Extract chunk slice from full array and pad with ghost borders.
        Ghosts outside the global domain are zero (Dirichlet BC).

        Parameters
        ----------
        arr : np.ndarray
            The full grid array from which to extract the chunk.
        r_start, r_end, c_start, c_end : int
            The start and end indices for the chunk slice in the full array.
        """
        rows, cols = self.full_shape
        chunk = np.zeros((self.chunk_x + 2, self.chunk_y + 2), dtype=arr.dtype)
        chunk[1:-1, 1:-1] = arr[r_start:r_end, c_start:c_end]

        if r_start > 0:  # Top edge
            chunk[0, 1:-1] = arr[r_start - 1, c_start:c_end]
        if r_end < rows:  # Bottom edge
            chunk[-1, 1:-1] = arr[r_end, c_start:c_end]
        if c_start > 0:   # Left edge
            chunk[1:-1, 0] = arr[r_start:r_end, c_start - 1]
        if c_end < cols:  # Right edge
            chunk[1:-1, -1] = arr[r_start:r_end, c_end]

        return chunk

    def scatter(self) -> dict:
        """
        Send each rank its local chunk parameters. Return rank 0's own parameters.
        """ 
        arrays_to_scatter = {
            "state": self.state,
            "domain_mask": self.domain_mask_full.astype(np.float64),
            "RHS": self.RHS.astype(np.float64),
            "theta_right": self.theta_right,
            "theta_left": self.theta_left,
            "theta_up": self.theta_up,
            "theta_down": self.theta_down,
            "MAX_ITER": self.MAX_ITER,
        }

        rank0_params = None

        for target_rank in range(self.n_chunks):
            r_start, r_end, c_start, c_end = self._get_chunk_slice(target_rank)
            loc = np.where(self.topology == target_rank)
            chunk_loc = (int(loc[0][0]), int(loc[1][0]))

            params = {
                "chunk_loc": chunk_loc,
                "topology": self.topology,
                "dimensions": (self.chunk_x, self.chunk_y),
                "omega": self.omega,
                "tol": self.tol,
                "dx": self.dx,
            }

            for name, full_arr in arrays_to_scatter.items():
                if isinstance(full_arr, np.ndarray):
                    chunk_arr = self._add_ghost_borders(full_arr, r_start, r_end, c_start, c_end)
                    params[name] = chunk_arr
                else:
                    params[name] = full_arr  # Scalar parameters like MAX_ITER

            # Fix mask back to bool after padding ghost borders
            params["domain_mask"] = params["domain_mask"].astype(bool)

            if target_rank == 0:
                rank0_params = params
            else:
                self.comm.send(params, dest=target_rank, tag=99)
        
        return rank0_params
    
    def reset(self, omega: float) -> dict:
        """
        Reset solution state and update omega for a new sweep.
        Avoids recomputing theta arrays.
        """
        self.omega = omega
        self.state = np.zeros_like(self.domain_mask_full, dtype=np.float64)
        self.residuals = []

        return self.scatter()
    
    def gather(self, chunk_array: np.ndarray):
        """
        Gather the partial chunks of chunk_array to root rank
        and reassemble the full array.
        Assumed that chunk_array is the local chunk of the full array 
        with ghost borders removed (i.e. shape (chunk_x, chunk_y)).
        """
        r_start, r_end, c_start, c_end = self._get_chunk_slice(0)
        r_buffer = np.empty_like(self.state, dtype=np.float64)

        r_buffer[r_start:r_end, c_start:c_end] = chunk_array

        for src_rank in range(1, self.n_chunks):
            data = self.comm.recv(source=src_rank, tag=100)
            r_start, r_end, c_start, c_end = self._get_chunk_slice(src_rank)
            r_buffer[r_start:r_end, c_start:c_end] = data  # Already removed when sent from chunk

        return r_buffer
    
    def poiseuille_coeff(self, chunk) -> float:
        """
        Compute C = dx^2 * sum(state) over interior points.
        For Poiseuille flow \nabal^2 u = -1, normalized by cross-sectional area.
        """
        self.state = self.gather(chunk)
        interior_sum = np.sum(self.state[self.domain_mask_full])
        area = np.sum(self.domain_mask_full) * self.dx**2

        return interior_sum * self.dx**2 / area

class SORChunk:
    """
    Worker chunk of the SOR grid. Receives its local slice from SORLattice via MPI.
    All ranks including 0 instantiate this — rank 0 gets its params returned directly
    from SORLattice.scatter(), all others receive via comm.recv(tag=99).
    """
    def __init__(self, 
                 comm: MPI.Comm,
                 params: dict = None,
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
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if params is None:
            params = self.comm.recv(source=0, tag=99)

        # Unpack topology and neighbor info
        self.topology  = params['topology']
        self.chunk_loc = params['chunk_loc']
        self.omega     = params['omega']
        self.tol       = params['tol']
        self.dx        = params['dx']
        self.zerodiv   = 1e-14
        self.MAX_ITER  = params.get('MAX_ITER', 10000)
        self.iter      = 0
        self.residuals = []
        self.verbose   = verbose
        self.very_verbose = False  # Internal flag for extra debug prints during communication steps
        self.bar = False

        # Neighbor lookup from topology
        if self.topology.ndim != 1:
            self.chunks = self.topology.shape[0]
            y, x = self.chunk_loc
            self.n_up    = self.topology[y - 1, x] if y > 0               else self.topology[-1, x]
            self.n_down  = self.topology[y + 1, x] if y < self.chunks - 1 else self.topology[0,  x]
            self.n_left  = self.topology[y, x - 1] if x > 0               else self.topology[y, -1]
            self.n_right = self.topology[y, x + 1] if x < self.chunks - 1 else self.topology[y,  0]
        else:
            self.chunks  = self.topology.shape[0]
            self.n_up    = self.topology[0]
            self.n_down  = self.topology[-1]
            self.n_left  = self.topology[0]
            self.n_right = self.topology[-1]
        self.n_neighbors = [self.n_up, self.n_down, self.n_left, self.n_right]

        # Unpack local arrays (all already include ghost ring from SORLattice._add_ghosts)
        self.state         = params['state'].astype(np.float64)
        self.init_state    = self.state.copy()
        self.domain_mask   = params['domain_mask'].astype(bool)
        self.theta_right   = params['theta_right']
        self.theta_left    = params['theta_left']
        self.theta_up      = params['theta_up']
        self.theta_down    = params['theta_down']
        self.RHS           = params['RHS']

        # GHOST HALO INCLUDED HERE 
        self.x_dim, self.y_dim = self.state.shape  

        if self.verbose:
            print(f"[bold green][Rank {self.rank}][/bold green] Ready! "
                  f"loc={self.chunk_loc} neighbors=({self.n_up},{self.n_down},{self.n_left},{self.n_right}) "
                  f"shape={self.state.shape}")

    def _broadcast_borders(self):
        """
        Broadcast the borders of the lattice to neighboring processes.
        Due to Checkerboard update we send all borders two times per step.

        Even ranks send first. Odd ranks receive first.
        First horizontal, then vertical communication.
        """
        y_loc, x_loc = self.chunk_loc
        if self.very_verbose:
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

        if self.very_verbose:
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
    
            if self.very_verbose:
                print(f"[bold green][Rank {self.rank}][/bold green] Waiting at barrier...")
        self.comm.Barrier()

    def _sor_step(self):
        """
        Perform one full SOR sweep: Red (parity 0) then Black (parity 1) half-steps with
        ghost border broadcasts in between.
        """
        self.prev_state = self.state.copy()
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
        delta = self.state - self.prev_state
        local_l2 = np.sum((delta[self.domain_mask])**2)

        # Switch to buffer-based Allreduce
        send_buf = np.array([local_l2], dtype=np.float64)
        recv_buf = np.zeroes(1, dtype=np.float64)
        self.comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)

        return float(np.sqrt(recv_buf[0]))

    def _global_residual_laplace(self):
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

    def run(self):
        """
        Run the SOR algorithm until convergence or max iterations. 
        """
        # Wait for all ranks to initialize
        self.comm.Barrier()

        if self.rank == 0 and self.verbose and self.bar:
            with Progress() as p:
                task = p.add_task(f"[bold green][All Ranks][/bold green]", total=self.MAX_ITER)
                while self.iter < self.MAX_ITER:
                    self._sor_step()
                    residual = self._global_residual()
                    self.residuals.append(residual)
                    self.iter += 1
                    p.update(task, advance=1, refresh=True, description=f"[bold green][All Ranks][/bold green] Iter {self.iter}, Residual {residual:.2e}")

                    if residual < self.tol:
                        print(f"[bold green][Rank 0][/bold green] Converged at iteration {self.iter}, residual {residual:.2e}")
                        break
        else:
            while self.iter < self.MAX_ITER:
                self._sor_step()
                residual = self._global_residual()
                self.residuals.append(residual)
                self.iter += 1
                if self.iter % 100 == 0 and self.rank == 0:
                    print(f"[bold green][Rank {self.rank}][/bold green] Iter {self.iter}, Residual {residual:.2e}", end='\r')
                
                if residual < self.tol:
                    break
        
        # Wait for all ranks to finish
        self.comm.Barrier()
        
        if self.verbose and self.rank == 0:
            print(f"[bold green][Rank {self.rank}][/bold green] Simulation Complete. " +  
                  f"Total Iterations: {self.iter}, Final Residual: {self.residuals[-1]:.2e}")
        
        # Truncate state to original dimensions (remove ghost rows and columns)
        self.state = np.ascontiguousarray(self.state[1:-1, 1:-1])
            
        # Send final state back to rank 0
        if self.rank != 0:
            self.comm.send(self.state, dest=0, tag=100)

        return self.state, self.residuals
