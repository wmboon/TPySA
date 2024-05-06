import numpy as np
import scipy.sparse as sps
import porepy as pp


class TPSA:
    def __init__(self) -> None:
        pass

    @staticmethod
    def dim_r(sd: pp.Grid):
        # Returns the dimension of the rotation space (d choose 2)
        return sd.dim * (sd.dim - 1) // 2

    def assemble(
        self, sd: pp.Grid, mu: np.ndarray, l2: np.ndarray, labda: np.ndarray
    ) -> sps.sparray:
        """
        Assemble the TPFA matrix, given material constants mu, l2, and lambda
        """
        # Save the unit normal vectors
        self.face_normals = sd.face_normals / sd.face_areas

        # Generate the matrices from (2.13) and (3.5)
        self.sigma = self.assemble_dual_var_map(sd, mu, l2)
        div = self.assemble_div(sd)

        A = div @ self.sigma
        M = self.mass(sd, mu, labda)

        return A - M

    def assemble_dual_var_map(
        self, sd: pp.Grid, mu: np.ndarray, l2: np.ndarray
    ) -> sps.sparray:
        """
        Assemble the matrix from (3.5) that maps primary to dual variables
        """
        # Extract cell-face pairs
        cf = sps.csc_array(sd.cell_faces)
        find_cf = sps.find(cf)

        self.delta_ki = self.assemble_delta_ki(sd, find_cf)

        # Assemble the blocks of (3.5) where A_ij is the block coupling variable i and j.
        mu_bar = self.harmonic_avg(find_cf, mu)
        A_uu = -2 * mu_bar[:, None] * cf
        A_uu = sps.block_diag([A_uu] * sd.dim)

        l2_bar = self.harmonic_avg(find_cf, l2)
        A_rr = -l2_bar[:, None] * cf
        A_rr = sps.block_diag([A_rr] * self.dim_r(sd))

        dk_mu = self.assemble_mu_dk(find_cf, mu, 2)
        A_pp = -dk_mu[:, None] * cf

        Xi = self.assemble_xi(find_cf, mu)
        Xi_tilde = self.assemble_xi_tilde(Xi)

        A_ru = self.assemble_S_Xi(sd, Xi, True)
        A_ur = self.assemble_S_Xi(sd, Xi_tilde, False)

        A_pr = self.assemble_n_Xi(sd, Xi, True)
        A_rp = self.assemble_n_Xi(sd, Xi_tilde, False)

        # Assembly by blocks
        # fmt: off
        A = sps.block_array(
            [[A_uu, A_ur, A_rp], 
             [A_ru, A_rr, None], 
             [A_pr, None, A_pp]]
        )
        # fmt: on

        # Scaling with the face areas
        face_areas = np.tile(sd.face_areas, sd.dim + self.dim_r(sd) + 1)

        return face_areas[:, None] * A

    def assemble_delta_ki(self, sd: pp.Grid, find_cf: tuple) -> np.ndarray:
        """
        Compute delta_k^i from (1.12) for every face-cell pair
        """
        faces, cells, orient = find_cf
        return np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * self.face_normals[:, faces])
            ),
            axis=0,
        )

    def harmonic_avg(self, find_cf: tuple, mu: np.ndarray) -> np.ndarray:
        """
        Compute the harmonic average of mu, divided by delta_k
        """

        # The numerator
        faces, cells, _ = find_cf
        mu_delta_ki = mu[cells] / self.delta_ki

        prod = sps.csc_array((mu_delta_ki, (cells, faces)))
        prod.sort_indices()
        numerator = prod.data[prod.indptr[:-1]] * prod.data[prod.indptr[1:] - 1]

        # The denominator
        denominator = self.assemble_mu_dk(find_cf, mu)

        return numerator * denominator

    def assemble_mu_dk(
        self, find_cf: tuple, mu: np.ndarray, alpha: float = 1
    ) -> np.ndarray:
        """
        Compute 1 / alpha( mu_i delta_k^-i + mu_j delta_k^-j)
        for each face k with cells (i,j)
        """
        faces, cells, _ = find_cf
        mu_delta_ki = mu[cells] / self.delta_ki

        return 1 / (alpha * np.bincount(faces, weights=mu_delta_ki))

    def assemble_xi(self, find_cf: tuple, mu: np.ndarray) -> sps.sparray:
        """
        Compute the averaging operator Xi
        """
        faces, cells, _ = find_cf
        Xi = sps.csc_array((mu[cells], (faces, cells)))
        Xi /= Xi.sum(axis=1)[:, None]

        return Xi

    def assemble_xi_tilde(self, Xi: sps.sparray) -> sps.sparray:
        """
        Compute the converse averaging operator Xi_tilde
        """
        Xi_tilde = Xi.copy()
        Xi_tilde.data = 1 - Xi_tilde.data

        return Xi_tilde

    def assemble_S_Xi(
        self, sd: pp.Grid, Xi: sps.sparray, u_to_r: bool = True
    ) -> sps.sparray:
        """
        Compute the adjoint of the asymmetry operator, acting on Xi
        """
        nx, ny, nz = [n_i[:, None] * Xi for n_i in self.face_normals]

        if sd.dim == 3:
            return -sps.block_array(
                [
                    [None, -nz, ny],
                    [nz, None, -nx],
                    [-ny, nx, None],
                ]
            )
        elif sd.dim == 2:
            if u_to_r:  # Maps from r to u
                return -sps.hstack([-ny, nx])
            else:  # Maps from r to u
                return -sps.vstack([ny, -nx])
        else:
            raise NotImplementedError("Dimension must be 2 or 3.")

    def assemble_n_Xi(
        self, sd: pp.Grid, Xi: sps.sparray, u_to_p: bool = True
    ) -> sps.sparray:
        """
        Normal times the averaging operator Xi
        """
        normal_times_xi = [n_i[:, None] * Xi for n_i in self.face_normals]

        if u_to_p:
            return sps.hstack(normal_times_xi[: sd.dim])
        else:  # Maps from r to u
            return sps.vstack(normal_times_xi[: sd.dim])

    def assemble_div(self, sd: pp.Grid) -> sps.sparray:
        """
        The divergence operator on the product space
        """
        dim = sd.dim + self.dim_r(sd) + 1
        return sps.block_diag([sd.cell_faces.T] * dim)

    def mass(self, sd: pp.Grid, mu: np.ndarray, labda: np.ndarray) -> sps.sparray:
        """
        The diagonal terms
        """
        M_u = sps.dia_array((sd.dim * sd.num_cells, sd.dim * sd.num_cells))
        M_r = sps.diags_array(np.tile(1 / mu, self.dim_r(sd)))
        M_p = sps.diags_array(1 / labda)

        M = sps.block_diag([M_u, M_r, M_p])
        cell_volumes = np.tile(sd.cell_volumes, sd.dim + self.dim_r(sd) + 1)

        return cell_volumes[:, None] * M
