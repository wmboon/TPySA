import numpy as np
import scipy.sparse as sps
import pygeon as pg


class TPSA:
    def __init__(self, sd, mu, l2) -> None:

        find_cf = sps.find(sd.cell_faces)

        self.delta_ki = self.assemble_delta_ki(sd, find_cf)
        # delta_k = np.bincount(faces, weights=delta_ki)

        mu_bar = self.harmonic_avg(find_cf, mu)
        l2_bar = self.harmonic_avg(find_cf, l2)
        dk_mu = self.assemble_dk_mu(find_cf, mu, 2)

        chi, chi_tilde = self.assemble_chis(find_cf, mu)

        pass

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

        Chi = sps.csc_array((mu, (faces, cells)))
        Chi /= Chi.sum(axis=1)[:, None]

        Chi_tilde = Chi.copy()
        Chi_tilde.data = 1 - Chi_tilde.data

        return Chi, Chi_tilde
