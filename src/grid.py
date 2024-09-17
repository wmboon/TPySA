import numpy as np
import scipy.sparse as sps

from opmcpg._cpggrid import (
    process_cpg_grid,
    value_vector,
    index_vector,
    UnstructuredGrid,
)
from opm.io.ecl import EclFile, EGrid


class Grid(EGrid):
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

    def __init__(self, egrid_file: str, result_fname="results.grdecl", minpv=0):
        super().__init__(egrid_file)

        grid_ecl = EclFile(egrid_file)

        # grid pillars, array of (nx+1)*(ny+1)*6 elements
        self.coord = grid_ecl["COORD"].astype(float)
        # grid nodes depths, array of nx*ny*nz*8 elements
        self.zcorn = grid_ecl["ZCORN"].astype(float)
        # integer array of nx*ny*nz elements, 0 - inactive cell, 1 - active cell
        self.actnum = grid_ecl["ACTNUM"]

        dims_cpp = index_vector(self.dimension)
        coord_cpp = value_vector(self.coord)
        zcorn_cpp = value_vector(self.zcorn)
        actnum_cpp = index_vector(self.actnum)

        unstr_grid = process_cpg_grid(
            dims_cpp, coord_cpp, zcorn_cpp, actnum_cpp, minpv, result_fname
        )

        self.dim = 3  # unstr_grid.dimensions

        self.num_cells = unstr_grid.number_of_cells
        self.num_faces = unstr_grid.number_of_faces
        self.num_nodes = unstr_grid.number_of_nodes

        self.face_normals = vectors_to_np(unstr_grid.face_normals)
        self.face_centers = vectors_to_np(unstr_grid.face_centroids)
        self.cell_centers = vectors_to_np(unstr_grid.cell_centroids)

        self.cell_volumes = scalars_to_np(unstr_grid.cell_volumes)
        self.face_areas = scalars_to_np(unstr_grid.face_areas)

        assert np.allclose(self.face_areas, np.linalg.norm(self.face_normals, axis=0))

        self.cell_faces = self.compute_cell_faces(unstr_grid)

    def compute_cell_faces(self, unstr_grid: UnstructuredGrid):
        face_cells = unstr_grid.face_cells[: self.num_faces * 2]
        face_cells = np.reshape(face_cells, (self.num_faces, 2))

        orientation = np.array([1, -1])
        I, J = np.nonzero(face_cells >= 0)

        return sps.csc_array((orientation[J], (I, face_cells[I, J])))


def vectors_to_np(input):
    return np.reshape(input, (3, -1), order="F", copy=False)


def scalars_to_np(input):
    return np.array(input, copy=False)
