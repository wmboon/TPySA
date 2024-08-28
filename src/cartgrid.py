import numpy as np
import scipy.sparse as sps
from opm.io.ecl import EGrid


class CartEGrid(EGrid):
    """
    Attributes:
        - dim
        - face_normals
        - face_areas
        - cell_faces
        - cell_centers
        - face_centers
        - cell_volumes
    """

    def __init__(self, egrid_file: str):
        """
        Creates a Grid object from a given egrid file

        Args:
            egrid_file (str): location of the egrid file
        """
        super().__init__(egrid_file)

        self.dim = len(self.dimension)
        self.nx, self.ny, self.nz = self.dimension
        self.num_cells = np.prod(np.array(self.dimension))
        self.cell_volumes = self.cellvolumes()

        lid_ind = self._find_lid_indices()

        self.cell_faces = self.compute_cell_faces(*lid_ind)
        self.num_faces = self.cell_faces.shape[0]

        xyz = self._find_corners_xyz()

        self.face_normals = self.compute_face_normals(xyz, *lid_ind)
        self.face_areas = self.compute_face_areas()

        self.cell_centers = self.compute_cell_centers(xyz)
        self.face_centers = self.compute_face_centers(xyz, *lid_ind)

    def _find_corners_xyz(self):
        """
        Returns a (num_cells, 3, 8) numpy array that contains the coordinates of the
        corner points of each grid cell
        """
        xyz = [self.xyz_from_active_index(k) for k in np.arange(self.active_cells)]
        return np.array(xyz)

    def _find_lid_indices(self):
        # Aliases
        nx, ny, nc = self.nx, self.ny, self.num_cells

        lid_x = np.arange(nx - 1, nc, nx)
        lid_y = np.arange(nx) + np.arange(nx * (ny - 1), nc + 1, nx * ny)[:, None]
        lid_y = lid_y.ravel()
        lid_z = np.arange(nc - nx * ny, nc)

        lid_cells = [lid_x, lid_y.ravel(), lid_z]

        lid_sizes = [len(lid) for lid in lid_cells]
        lid_faces = 3 * self.num_cells + np.arange(np.sum(lid_sizes))

        lid_faces = np.split(lid_faces, np.cumsum(lid_sizes))[:-1]

        return lid_cells, lid_faces

    def compute_cell_faces(self, lid_c, lid_f):
        # Cell-face pairs
        cells = np.tile(np.arange(self.num_cells), (6, 1))

        faces = np.empty((6, self.num_cells), int)
        faces[:3, :] = np.reshape(np.arange(self.num_cells * 3), (3, -1), order="C")

        offset = np.array([1, self.nx, self.nx * self.ny])[:, None]
        faces[3:, :] = faces[:3, :] + offset

        # Take care of the lids
        for i in np.arange(3):
            faces[i + 3, lid_c[i]] = lid_f[i]

        # Orientations
        orien = np.ones_like(faces)
        orien[:3, :] = -1

        return sps.csc_array((orien.ravel(), (faces.ravel(), cells.ravel())))

    def compute_face_areas(self):
        return np.linalg.norm(self.face_normals, axis=0)

    def compute_cell_centers(self, xyz):
        return np.mean(xyz, axis=2).T

    def compute_face_normals(self, xyz, lid_c, lid_f):
        fn = np.empty((self.dim, self.num_faces))

        def cross(xyz, i, j, k=0):
            return np.cross(xyz[:, :, i] - xyz[:, :, k], xyz[:, :, j] - xyz[:, :, k]).T

        # i-, j-, and k- faces
        fn[:, 0 * self.num_cells : 1 * self.num_cells] = cross(xyz, 2, 4)
        fn[:, 1 * self.num_cells : 2 * self.num_cells] = cross(xyz, 4, 1)
        fn[:, 2 * self.num_cells : 3 * self.num_cells] = cross(xyz, 1, 2)

        # i+, j+, and k+ lids
        fn[:, lid_f[0]] = cross(xyz[lid_c[0], :, :], 3, 5, 1)
        fn[:, lid_f[1]] = cross(xyz[lid_c[1], :, :], 6, 3, 2)
        fn[:, lid_f[2]] = cross(xyz[lid_c[2], :, :], 5, 6, 4)

        return fn

    def compute_face_centers(self, xyz, lid_c, lid_f):
        fc = np.empty((self.dim, self.num_faces))

        # We define the face center as the average of the adjacent node coordinates
        def avg(xyz, indices):
            return np.mean(xyz[:, :, indices], axis=-1).T

        # i-, j-, and k- faces
        fc[:, 0 * self.num_cells : 1 * self.num_cells] = avg(xyz, [0, 2, 4, 6])
        fc[:, 1 * self.num_cells : 2 * self.num_cells] = avg(xyz, [0, 1, 4, 5])
        fc[:, 2 * self.num_cells : 3 * self.num_cells] = avg(xyz, [0, 1, 2, 3])

        # i+, j+, and k+ lids
        fc[:, lid_f[0]] = avg(xyz[lid_c[0], :, :], [1, 3, 5, 7])
        fc[:, lid_f[1]] = avg(xyz[lid_c[1], :, :], [2, 3, 6, 7])
        fc[:, lid_f[2]] = avg(xyz[lid_c[2], :, :], [4, 5, 6, 7])

        return fc
