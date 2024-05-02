import numpy as np
import scipy.sparse as sps
import pygeon as pg


class TPSA:
    def __init__(self, sd, mu, l2) -> None:

        find_cf = sps.find(sd.cell_faces)

        self.dki = self.assemble_delta_ki(sd, find_cf)

        mu_hat, mu_bar = self.assemble_hat_bar(find_cf, mu)
        l2_hat, l2_bar = self.assemble_hat_bar(find_cf, l2)

    def assemble_delta_ki(self, sd, find_cf):
        faces, cells, orient = find_cf
        return np.sum(
            (
                (sd.face_centers[:, faces] - sd.cell_centers[:, cells])
                * (orient * sd.face_normals[:, faces])
            ),
            axis=0,
        )

    def assemble_hat_bar(self, find_cf, mu):
        faces, cells, _ = find_cf
        # delta_k = np.bincount(faces, weights=delta_ki)
        # delta_bar_k = np.bincount(faces, weights=1.0 / delta_ki)

        mu_delta_ki = mu[cells] / self.delta_ki

        mu_hat = np.bincount(faces, weights=mu_delta_ki)

        # Product in the numerator
        prod = sps.csr_array((mu, (faces, cells)))
        prod.sort_indices()

        mu_bar = (
            prod.data[prod.indptr[:-1]] * prod.data[prod.indptr[1:] - 1] / self.mu_hat
        )

        return mu_hat, mu_bar

    def assemble_chi(self, find_cf, mu):
        faces, cells, _ = find_cf

        Chi = sps.csr_array((mu, (faces, cells)))
        Chi /= Chi.sum(axis=1)[:, None]
        Chi_tilde = Chi.copy()
        Chi_tilde.data = 1 - Chi_tilde.data
