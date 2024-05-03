import numpy as np
import scipy.sparse as sps


class TPSA:
    def __init__(self) -> None:
        pass

    def dim_r(self, sd):
        return sd.dim * (sd.dim - 1) // 2

    def assemble(self, sd, mu, l2, labda):

        sigma = self.primary_to_dual_map(sd, mu, l2)
        div = self.div_map(sd)

        A = div @ sigma

        M = self.mass(sd, mu, labda)

        return A - M

    def primary_to_dual_map(self, sd, mu, l2):
        cf = sps.csc_array(sd.cell_faces)
        find_cf = sps.find(cf)

        self.delta_ki = self.assemble_delta_ki(sd, find_cf)
        # delta_k = np.bincount(faces, weights=delta_ki)

        mu_bar = self.harmonic_avg(find_cf, mu)
        A_uu = -2 * mu_bar[:, None] * cf
        A_uu = sps.block_diag([A_uu] * sd.dim)

        l2_bar = self.harmonic_avg(find_cf, l2)
        A_rr = -l2_bar[:, None] * cf
        A_rr = sps.block_diag([A_rr] * self.dim_r(sd))

        dk_mu = self.assemble_dk_mu(find_cf, mu, 2)
        A_pp = -dk_mu[:, None] * cf

        chi, chi_tilde = self.assemble_chis(find_cf, mu)

        A_ru = self.assemble_S_star(sd, chi, True)
        A_ur = self.assemble_S_star(sd, chi_tilde, False)

        A_pr = self.assemble_n_chi(sd, chi, True)
        A_rp = self.assemble_n_chi(sd, chi_tilde, False)

        return sps.block_array(
            [[A_uu, A_ur, A_rp], [A_ru, A_rr, None], [A_pr, None, A_pp]]
        )

    def assemble_delta_ki(self, sd, find_cf):
        faces, cells, orient = find_cf
        return np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * sd.face_normals[:, faces] / sd.face_areas[faces])
            ),
            axis=0,
        )

    def harmonic_avg(self, find_cf, mu):
        faces, cells, _ = find_cf
        mu_delta_ki = mu[cells] / self.delta_ki

        mu_hat = self.assemble_dk_mu(find_cf, mu)

        # Product in the numerator
        prod = sps.csc_array((mu_delta_ki, (cells, faces)))
        prod.sort_indices()

        mu_bar = prod.data[prod.indptr[:-1]] * prod.data[prod.indptr[1:] - 1] * mu_hat

        return mu_bar

    def assemble_dk_mu(self, find_cf, mu, alpha=1):
        faces, cells, _ = find_cf
        # delta_bar_k = np.bincount(faces, weights=1.0 / delta_ki)

        mu_delta_ki = mu[cells] / self.delta_ki

        return 1 / (alpha * np.bincount(faces, weights=mu_delta_ki))

    def assemble_chis(self, find_cf, mu):
        faces, cells, _ = find_cf

        Chi = sps.csc_array((mu[cells], (faces, cells)))
        Chi /= Chi.sum(axis=1)[:, None]

        Chi_tilde = Chi.copy()
        Chi_tilde.data = 1 - Chi_tilde.data

        return Chi, Chi_tilde

    def assemble_S_star(self, sd, chi, u_to_r=True):
        nx, ny, nz = [n_i[:, None] for n_i in sd.face_normals]

        if sd.dim == 3:
            return -sps.block_array(
                [
                    [None, -nz * chi, ny * chi],
                    [nz * chi, None, -nx * chi],
                    [-ny * chi, nx * chi, None],
                ],
                format="csc",
            )
        elif sd.dim == 2:
            if u_to_r:
                return -sps.hstack(
                    [-ny * chi, nx * chi],
                    format="csc",
                )
            else:
                return -sps.vstack(
                    [ny * chi, -nx * chi],
                    format="csc",
                )
        else:
            raise NotImplementedError("Dimension must be 2 or 3.")

    def assemble_n_chi(self, sd, chi, u_to_p=True):
        normals = [n_i[:, None] for n_i in sd.face_normals]
        normal_times_chi = [n_i * chi for n_i in normals[: sd.dim]]

        if u_to_p:
            return sps.hstack(normal_times_chi, format="csc")
        else:
            return sps.vstack(normal_times_chi, format="csc")

    def div_map(self, sd):
        dim = sd.dim + self.dim_r(sd) + 1
        return sps.block_diag([sd.cell_faces.T] * dim, format="csc")

    def mass(self, sd, mu, labda):
        M_u = sps.csc_array((sd.dim * sd.num_cells, sd.dim * sd.num_cells))
        M_r = sps.diags_array(np.tile(sd.cell_volumes / mu, self.dim_r(sd)))
        M_p = sps.diags_array(sd.cell_volumes / labda)

        return sps.block_diag([M_u, M_r, M_p])
