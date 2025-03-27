import numpy as np
import scipy.sparse as sps
import time
import tpysa


class TPSA:
    def __init__(self, grid: tpysa.Grid):
        """
        Initializes an instance of the TPSA discretization on a given grid.
        """

        # Save the grid
        self.sd = grid

        # Save the dimension of the rotation space (d choose 2)
        self.dim_r = grid.dim * (grid.dim - 1) // 2

        # Numbers of degrees of freedom for the displacement, rotation, and pressure
        self.ndof_per_cell = self.sd.dim + self.dim_r + 1
        self.ndofs = grid.num_cells * np.array([grid.dim, self.dim_r, 1])

        # Save the unit normal vectors
        self.unit_normals = self.sd.face_normals / self.sd.face_areas

        # Save the cell_face connectivity
        self.find_cf = sps.find(self.sd.cell_faces)

    def discretize(self, data: dict) -> sps.sparray:
        """
        Assemble the TPSA matrix, given material constants in the data dictionary.
        Also sets the reference pressure, if available.
        """
        start_time = time.time()

        # Extract the mean of the second Lamé parameter
        data["scaling"] = np.mean(data["mu"])

        # Generate the matrices from (2.13) and (3.5)
        self.sigma = self.assemble_dual_var_map(data, scale_factor=data["scaling"])
        div = self.assemble_div()

        A = div @ self.sigma
        M = self.mass(data, scale_factor=data["scaling"])

        self.system = sps.csc_array(A - M)
        print("TPSA: Assembled system ({:.2f} sec)".format(time.time() - start_time))

        if "ref_pressure" in data:
            self.ref_pressure = data["ref_pressure"]
        else:
            print("WARNING: no reference pressure given")
            self.ref_pressure = np.zeros(self.sd.num_cells)

    def assemble_dual_var_map(
        self, data: dict, scale_factor: float = 1.0
    ) -> sps.sparray:
        """
        Assemble the matrix from (3.7) that maps primary to dual variables
        """
        # Extract cell-face pairs
        cf = sps.csc_array(self.sd.cell_faces)

        self.mu_delta_ki = self.assemble_mu_delta_ki(data["mu"])

        # Assemble the blocks of (3.7) where
        # A_ij is the block coupling variable i and j.
        mu_bar = self.harmonic_avg()
        A_uu = -2 * mu_bar[:, None] / scale_factor * cf
        A_uu = sps.block_diag([A_uu] * self.sd.dim)

        dk_mu = self.assemble_delta_mu_k()
        A_pp = -dk_mu[:, None] * scale_factor * cf

        Xi = self.assemble_xi()
        Xi_tilde = self.assemble_xi_tilde(Xi)

        A_ru = self.assemble_R_Xi(Xi, True)
        A_ur = self.assemble_R_Xi(Xi_tilde, False)

        A_pr = self.assemble_n_Xi(Xi, True)
        A_rp = self.assemble_n_Xi(Xi_tilde, False)

        # Assembly by blocks
        # fmt: off
        A = sps.block_array(
            [[A_uu, A_ur, A_rp], 
             [A_ru, None, None], 
             [A_pr, None, A_pp]]
        )
        # fmt: on

        # Scaling with the face areas
        face_areas = np.tile(self.sd.face_areas, self.ndof_per_cell)
        A = face_areas[:, None] * A

        return A

    def assemble_mu_delta_ki(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute mu / delta_k^i from (1.12) for every face-cell pair
        """
        faces, cells, orient = self.find_cf
        delta_ki = np.sum(
            (
                (self.sd.face_centers[:, faces] - self.sd.cell_centers[:, cells])
                * (orient * self.unit_normals[:, faces])
            ),
            axis=0,
        )
        return mu[cells] / delta_ki

    def harmonic_avg(self) -> np.ndarray:
        """
        Compute the harmonic average of mu divided by delta_k
        """

        # The numerator
        # for interior faces, it takes the product of the two mu / delta
        # for boundary faces, it computes (mu / delta)^2

        faces, cells, _ = self.find_cf

        prod = sps.csc_array((self.mu_delta_ki, (cells, faces)))
        prod.sort_indices()
        numerator = prod.data[prod.indptr[:-1]] * prod.data[prod.indptr[1:] - 1]

        # The denominator
        # for interior faces, it takes the sum of mu / delta
        # for boundary faces, it computes mu / delta

        denominator = np.bincount(faces, weights=self.mu_delta_ki)

        return numerator / denominator

    def assemble_delta_mu_k(self) -> np.ndarray:
        """
        Compute 1 / 2 ( mu_i delta_k^-i + mu_j delta_k^-j)
        for each face k with cells (i,j)
        """

        # Interior faces
        faces, _, _ = self.find_cf
        dk_mu = 1 / (2 * np.bincount(faces, weights=self.mu_delta_ki))

        # Boundary faces with displacement conditions
        dk_mu[self.sd.tags["displ_bdry"]] = 0

        return dk_mu

    def assemble_xi(self) -> sps.sparray:
        """
        Compute the averaging operator Xi
        """
        faces, cells, _ = self.find_cf
        Xi = sps.csc_array((self.mu_delta_ki, (faces, cells)))

        Xi *= (np.logical_not(self.sd.tags["displ_bdry"]) / Xi.sum(axis=1))[:, None]

        return Xi

    def assemble_xi_tilde(self, Xi: sps.sparray) -> sps.sparray:
        """
        Compute the converse averaging operator Xi_tilde
        """
        Xi_tilde = Xi.copy()
        Xi_tilde.data = 1 - Xi_tilde.data

        return Xi_tilde

    def assemble_R_Xi(self, Xi: sps.sparray, u_to_r: bool = True) -> sps.sparray:
        """
        Compute the adjoint of the asymmetry operator, acting on Xi
        """
        nx, ny, nz = [n_i[:, None] * Xi for n_i in self.unit_normals]

        if self.sd.dim == 3:
            return -sps.block_array(
                [
                    [None, -nz, ny],
                    [nz, None, -nx],
                    [-ny, nx, None],
                ]
            )
        elif self.sd.dim == 2:
            if u_to_r:  # Maps from u to r
                return -sps.hstack([-ny, nx])
            else:  # Maps from r to u
                return -sps.vstack([ny, -nx])
        else:
            raise NotImplementedError("Dimension must be 2 or 3.")

    def assemble_n_Xi(self, Xi: sps.sparray, u_to_p: bool = True) -> sps.sparray:
        """
        Normal times the averaging operator Xi
        """
        normal_times_xi = [n_i[:, None] * Xi for n_i in self.unit_normals]

        if u_to_p:
            return sps.hstack(normal_times_xi[: self.sd.dim])
        else:  # Maps from r to u
            return sps.vstack(normal_times_xi[: self.sd.dim])

    def assemble_div(self) -> sps.sparray:
        """
        The divergence operator on the product space
        """
        return sps.block_diag([self.sd.cell_faces.T.tocsc()] * self.ndof_per_cell)

    def mass(self, data: dict, scale_factor: float = 1.0) -> sps.sparray:
        """
        The diagonal terms
        """
        M_u = sps.dia_array((self.ndofs[0], self.ndofs[0]))
        M_r = sps.diags_array(np.tile(scale_factor / data["mu"], self.dim_r))
        M_p = sps.diags_array(scale_factor / data["lambda"])

        M = sps.block_diag([M_u, M_r, M_p])
        cell_volumes = np.tile(self.sd.cell_volumes, self.ndof_per_cell)

        return cell_volumes[:, None] * M

    def assemble_isotropic_stress_source(self, data: dict, w: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side for a given isotropic stress field w, like a fluid pressure
        """
        rhs_u = np.zeros(self.ndofs[0])
        rhs_r = np.zeros(self.ndofs[1])
        rhs_p = self.sd.cell_volumes * data["alpha"] / data["lambda"] * w

        return np.hstack((rhs_u, rhs_r, rhs_p))

    def assemble_gravity_force(self, data: dict) -> np.ndarray:
        w = np.zeros(self.ndofs[0])
        indices_uz = np.arange(
            (self.sd.dim - 1) * self.sd.num_cells, self.sd.dim * self.sd.num_cells
        )
        w[indices_uz] = data["gravity"]

        return self.assemble_body_force(w)

    def assemble_body_force(self, f: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side for a given body force f(x,y,z)
        We assume that these are numbered as
            f[0]   = f_x(x_0)
            f[1]   = f_x(x_1)
            ...
            f[n_c] = f_y(x_0)
            ...
        """
        rhs_u = np.tile(self.sd.cell_volumes, self.sd.dim) * f
        rhs_r = np.zeros(self.ndofs[1])
        rhs_p = np.zeros(self.ndofs[2])

        return np.hstack((rhs_u, rhs_r, rhs_p))

    def solve(
        self, data: dict, pressure_source: np.ndarray, solver: tpysa.Solver
    ) -> tuple[np.ndarray]:
        diff_pressure = pressure_source - self.ref_pressure
        rhs = self.assemble_isotropic_stress_source(data, diff_pressure)

        scale_factor = np.hstack(
            (
                np.full(self.ndofs[0], 1 / np.sqrt(data["scaling"])),
                np.full(self.ndofs[1] + self.ndofs[2], np.sqrt(data["scaling"])),
            )
        )
        rhs *= scale_factor

        sol, info = solver.solve(rhs)
        assert info == 0

        sol *= scale_factor
        u, r, p, _ = np.split(sol, np.cumsum(self.ndofs))

        return u, r, p

    def recover_volumetric_change(
        self, solid_p: np.ndarray, fluid_p: np.ndarray, data: dict
    ) -> np.ndarray:
        return (solid_p + data["alpha"] * (fluid_p - self.ref_pressure)) / data[
            "lambda"
        ]
