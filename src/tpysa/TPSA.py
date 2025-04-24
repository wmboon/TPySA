import logging
import time

import numpy as np
import scipy.sparse as sps

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
        normal_norms = np.linalg.norm(self.sd.face_normals, axis=0)
        self.unit_normals = self.sd.face_normals / normal_norms

        # Save the cell_face connectivity
        self.find_cf = sps.find(self.sd.cell_faces)

    def discretize(self, data: dict) -> None:
        """
        Assemble the TPSA matrix, given material constants in the data dictionary.
        Also sets the reference pressure, if available.
        """
        start_time = time.time()

        # The mean of mu is used to scale the resulting system
        data["scaling"] = np.mean(data["mu"])

        # Extract the spring constant
        self.bdry_mu_delta = data.get("bdry_mu_over_delta", np.zeros(self.sd.num_faces))

        # In order to allow for a different scaling, we precompute mu/dk here
        self.mu_delta_ki = self.assemble_mu_delta_ki(data["mu"])
        self.dk_mu = self.assemble_delta_mu_k()

        # Generate the matrices from (2.13) and (3.5)
        A = self.assemble_second_order_terms(scale_factor=data["scaling"])
        M = self.assemble_first_order_terms(data, scale_factor=data["scaling"])

        self.system = sps.csc_array(A - M)
        logging.info(
            "Assembled system with {:} dof ({:.2f} sec)".format(
                self.system.shape[0], time.time() - start_time
            )
        )

        if "ref_pressure" in data:
            self.ref_pressure = data["ref_pressure"]
        else:
            logging.warning("No reference pressure given; setting p_ref to zero.")
            self.ref_pressure = np.zeros(self.sd.num_cells)

    def assemble_second_order_terms(self, scale_factor: float = 1.0) -> sps.sparray:
        """
        Assemble the matrix from (3.7) that maps primary to dual variables
        """
        # Precompute the operator that multiplies with the area
        # and applies the divergence
        div_F = sps.csc_array(self.sd.cell_faces.T) * self.sd.face_areas

        # Preallocate the main matrix
        A = np.empty((3, 3), dtype=sps.sparray)

        # Assemble the blocks of (3.7) where
        # A_ij is the block coupling variable i and j.
        mu_bar = self.harmonic_avg()
        A_uu = -2 * div_F * (mu_bar / scale_factor) @ self.sd.cell_faces
        A[0, 0] = sps.block_diag([A_uu] * self.sd.dim)

        # The blocks in the first column depend on the averaging operator Xi
        Xi = self.assemble_xi()
        A[1, 0], A[2, 0] = self.assemble_off_diagonals(Xi, div_F, True)

        # The blocks in the first row depend on the complementary operator Xi_tilde
        Xi_tilde = self.convert_to_xi_tilde(Xi)
        A[0, 1], A[0, 2] = self.assemble_off_diagonals(Xi_tilde, div_F, False)

        A[2, 2] = -div_F * (0.5 * self.dk_mu * scale_factor) @ self.sd.cell_faces

        # Assembly by blocks
        return sps.block_array(A).tocsc()

    def assemble_mu_delta_ki(self, mu: np.ndarray) -> np.ndarray:
        """
        Compute mu / delta_k^i from (1.12) for every physical face-cell pair.
        Boundary conditions are handled later
        """
        faces, cells, orient = self.find_cf

        def compute_delta_ki(indices=slice(None)):
            return np.sum(
                (
                    (
                        self.sd.face_centers[:, faces[indices]]
                        - self.sd.cell_centers[:, cells[indices]]
                    )
                    * (orient[indices] * self.unit_normals[:, faces[indices]])
                ),
                axis=0,
            )

        delta_ki = compute_delta_ki()

        # Check if any cell centers are placed outside the cell
        if np.any(delta_ki < 0):
            logging.debug(
                "Moving {} extra-cellular centers to the mean of the nodes".format(
                    np.sum(delta_ki < 0)
                )
            )
            for cell in cells[delta_ki <= 0]:
                cf_pairs = cells == cell

                # Define a cell-center based on the mean of the 8 nodes
                xyz = self.sd.xyz_from_active_index(cell)
                self.sd.cell_centers[:, cell] = np.mean(xyz, axis=1)

                # Recompute the deltas with the updated cell center
                delta_ki[cf_pairs] = compute_delta_ki(cf_pairs)

            if np.any(delta_ki < 0):
                # Report on the first problematic cell for visual inspection
                first_cell = cells[np.argmax(delta_ki <= 0)]
                ijk = self.sd.ijk_from_active_index(first_cell)
                glob_ind = self.sd.global_index(*ijk)

                logging.debug(
                    "{} extra-cellular centers remain".format(np.sum(delta_ki < 0))
                )
                logging.warning(
                    "Cell with global index {} has an extra-cellular center.\n".format(
                        glob_ind
                    )
                )
            else:
                logging.debug("Fixed all cell-centers\n")

        return mu[cells] / delta_ki

    def assemble_delta_mu_k(self) -> np.ndarray:
        """
        Compute ( mu_i delta_k^-i + mu_j delta_k^-j)^-1
        for each face k with cells (i,j)
        """

        faces, _, _ = self.find_cf
        mu_dk = self.mu_delta_ki

        # Spring bc
        if np.any(self.sd.tags["sprng_bdry"]):
            faces = np.concatenate((faces, np.flatnonzero(self.sd.tags["sprng_bdry"])))
            mu_dk = np.concatenate(
                (mu_dk, self.bdry_mu_delta[self.sd.tags["sprng_bdry"]])
            )

        dk_mu = 1 / (np.bincount(faces, weights=mu_dk))

        # Displacement bc
        dk_mu[self.sd.tags["displ_bdry"]] = 0

        # Traction bc are handled naturally as a subset of spring_bdry
        # with zero spring constant

        return dk_mu

    def harmonic_avg(self) -> np.ndarray:
        """
        Compute the harmonic average of mu, divided by delta_k, at each face
        """

        # The numerator
        #   for interior faces, it takes the product of the two mu / delta
        #   for boundary faces, it computes (mu / delta)^2

        faces, cells, _ = self.find_cf

        prod = sps.csc_array((self.mu_delta_ki, (cells, faces)))
        prod.sort_indices()
        numerator = prod.data[prod.indptr[:-1]] * prod.data[prod.indptr[1:] - 1]

        # The denominator
        #   for interior faces, it takes the sum of the two mu / delta
        #   for boundary faces, it computes mu / delta

        denominator = np.bincount(faces, weights=self.mu_delta_ki)

        mu_bar_delta = numerator / denominator

        # Displacement bc are handled naturally

        # Spring bc
        mask = self.sd.tags["sprng_bdry"]
        mu_bar_delta[mask] *= self.bdry_mu_delta[mask] / (
            self.bdry_mu_delta[mask] + mu_bar_delta[mask]
        )

        # This also takes care of traction bc because bdry_mu_delta is zero there

        return mu_bar_delta

    def assemble_xi(self) -> sps.sparray:
        """
        Compute the averaging operator Xi
        """
        faces, cells, _ = self.find_cf
        Xi = sps.csc_array((self.mu_delta_ki * self.dk_mu[faces], (faces, cells)))

        # Displacement bc are handled by dk_mu = 0
        # Traction bc are handled since dk_mu * mu_dk = 1
        # Spring bc are handled because the spring constant is in dk_mu

        return Xi

    def convert_to_xi_tilde(self, Xi: sps.sparray) -> sps.sparray:
        """
        Compute the converse averaging operator Xi_tilde
        This is an in-place operation to limit memory
        """
        Xi.data = 1 - Xi.data
        return Xi

    def assemble_off_diagonals(
        self,
        Xi: sps.sparray,
        div_F: sps.sparray,
        map_from_u: bool = True,
    ) -> tuple[sps.sparray, sps.sparray]:
        if self.sd.dim == 3:
            nx, ny, nz = [(div_F * ni) @ Xi for ni in self.unit_normals]

            R_Xi = -sps.block_array(
                [
                    [None, -nz, ny],
                    [nz, None, -nx],
                    [-ny, nx, None],
                ]
            )

            if map_from_u:  # Maps from u to p
                n_Xi = sps.hstack([nx, ny, nz])
            else:  # Maps from p to u
                n_Xi = sps.vstack([nx, ny, nz])

            return R_Xi, n_Xi
        else:
            raise NotImplementedError

    def assemble_first_order_terms(
        self, data: dict, scale_factor: float = 1.0
    ) -> sps.csc_array:
        """
        The diagonal terms
        """
        volumes = self.sd.cell_volumes
        M_u = np.zeros(self.ndofs[0])
        M_r = np.tile(scale_factor * volumes / data["mu"], self.dim_r)
        M_p = scale_factor * volumes / data["lambda"]

        diagonal = np.concatenate((M_u, M_r, M_p))

        return sps.diags_array(diagonal).tocsc()

    def assemble_isotropic_stress_source(self, data: dict, w: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side for a given isotropic stress field w,
        like a fluid pressure
        """
        rhs_u = np.zeros(self.ndofs[0])
        rhs_r = np.zeros(self.ndofs[1])
        rhs_p = self.sd.cell_volumes * data["alpha"] / data["lambda"] * w

        return np.concatenate((rhs_u, rhs_r, rhs_p))

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

        return np.concatenate((rhs_u, rhs_r, rhs_p))

    def solve(
        self, data: dict, pressure_source: np.ndarray, solver: tpysa.Solver
    ) -> tuple[np.ndarray]:
        diff_pressure = pressure_source - self.ref_pressure
        rhs = self.assemble_isotropic_stress_source(data, diff_pressure)

        scale_factor = np.concatenate(
            (
                np.full(self.ndofs[0], 1 / np.sqrt(data["scaling"])),
                np.full(self.ndofs[1] + self.ndofs[2], np.sqrt(data["scaling"])),
            )
        )
        rhs *= scale_factor

        rtol = data.get("rtol", 1e-5)
        sol, info = solver.solve(rhs, rtol=rtol)
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

    # ------------------------------- DEPRECATED -------------------------------
    def assemble_dual_var_map(self, scale_factor: float = 1.0) -> sps.sparray:
        """
        Assemble the matrix from (3.7) that maps primary to dual variables
        This function is only used for post-processing the stress
        """
        # Extract cell-face pairs
        cf = sps.csc_array(self.sd.cell_faces)

        # Assemble the blocks of (3.7) where
        # A_ij is the block coupling variable i and j.
        mu_bar = self.harmonic_avg()
        A_uu = -2 * mu_bar[:, None] / scale_factor * cf
        A_uu = sps.block_diag([A_uu] * self.sd.dim)

        dk_mu = self.assemble_delta_mu_k()
        A_pp = -dk_mu[:, None] * scale_factor * cf

        Xi = self.assemble_xi()
        Xi_tilde = self.convert_to_xi_tilde(Xi)

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

    def assemble_R_Xi(
        self,
        Xi: sps.sparray,
        u_to_r: bool = True,
        div=None,
    ) -> sps.sparray:
        """
        Compute the adjoint of the asymmetry operator, acting on Xi
        """
        if div is None:
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
        else:

            def n(x: int):
                return (div * self.unit_normals[x]) @ Xi

            return -sps.block_array(
                [
                    [None, -n(2), n(1)],
                    [n(2), None, -n(0)],
                    [-n(1), n(0), None],
                ]
            )

    def assemble_n_Xi(
        self,
        Xi: sps.sparray,
        u_to_p: bool = True,
        div=None,
    ) -> sps.sparray:
        """
        Normal times the averaging operator Xi
        """
        if div is None:
            normal_times_xi = [n_i[:, None] * Xi for n_i in self.unit_normals]
        else:
            normal_times_xi = [div @ (n_i[:, None] * Xi) for n_i in self.unit_normals]

        if u_to_p:
            return sps.hstack(normal_times_xi[: self.sd.dim])
        else:  # Maps from p to u
            return sps.vstack(normal_times_xi[: self.sd.dim])

    def assemble_div(self) -> sps.sparray:
        """
        The divergence operator on the product space
        """
        return sps.block_diag([self.sd.cell_faces.T.tocsc()] * self.ndof_per_cell)
