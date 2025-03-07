import numpy as np
import scipy.sparse as sps

from opmcpg._cpggrid import (
    process_cpg_grid,
    value_vector,
    index_vector,
    UnstructuredGrid,
)
from opm.io.ecl import EclFile, EGrid
import vtk


class Grid(EGrid):
    """
    Attributes required by TPSA:
        - dim
        - cell_centers
        - cell_volumes
        - face_centers
        - face_normals
        - face_areas
        - cell_faces

    Attributes further desired by PorePy and therefore implemented here:
        - num_cells/num_faces/num_nodes
        - nodes
        - face_nodes
        - tags["domain_boundary_faces"]
    """

    def __init__(self, egrid_file: str, result_fname="results.grdecl", minpv=0):
        super().__init__(egrid_file)

        grid_ecl = EclFile(egrid_file)

        # Grid pillars, array of (nx+1)*(ny+1)*6 elements
        self.coord = grid_ecl["COORD"].astype(float)
        # Grid nodes depths, array of nx*ny*nz*8 elements
        self.zcorn = grid_ecl["ZCORN"].astype(float)
        # Integer array of nx*ny*nz elements, 0 - inactive cell, 1 - active cell
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
        self.nodes = vectors_to_np(unstr_grid.node_coordinates)

        self.cell_volumes = scalars_to_np(unstr_grid.cell_volumes)
        self.face_areas = scalars_to_np(unstr_grid.face_areas)

        self.cell_faces = self.compute_cell_faces(unstr_grid)
        self.face_nodes = self.compute_face_nodes(unstr_grid)

        self.tags = {}
        self.tag_boundaries()

    def compute_face_nodes(self, unstr_grid: UnstructuredGrid) -> sps.csc_array:
        indptr = scalars_to_np(unstr_grid.face_nodepos)[: self.num_faces + 1]
        indices = scalars_to_np(unstr_grid.face_nodes)[: indptr[-1]]
        data = np.ones_like(indices, dtype=bool)

        return sps.csc_array((data, indices, indptr))

    def compute_cell_faces(self, unstr_grid: UnstructuredGrid) -> sps.csc_array:
        face_cells = unstr_grid.face_cells[: self.num_faces * 2]
        face_cells = np.reshape(face_cells, (self.num_faces, 2))

        orientation = np.array([1, -1])
        face_row, cell_col = np.nonzero(face_cells >= 0)

        return sps.csc_array(
            (orientation[cell_col], (face_row, face_cells[face_row, cell_col]))
        )

    def compute_cell_nodes(self) -> sps.csc_array:
        return self.face_nodes @ self.cell_faces.astype(bool)

    def tag_boundaries(self) -> None:
        num_cells_per_face = self.cell_faces.sum(axis=1)
        self.tags["domain_boundary_faces"] = num_cells_per_face != 0
        self.tags["domain_boundary_nodes"] = (
            self.face_nodes @ self.tags["domain_boundary_faces"]
        )

    def get_vtk(self) -> vtk.vtkUnstructuredGrid:
        if not hasattr(self, "vtk_grid"):
            cell_nodes = self.compute_cell_nodes()

            points = vtk.vtkPoints()
            for i, x in enumerate(self.nodes.T):
                points.InsertPoint(i, x)

            self.vtk_grid = vtk.vtkUnstructuredGrid()
            self.vtk_grid.Allocate(self.num_cells)

            for cell in np.arange(self.num_cells):
                node_inds = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
                nodes = cell_nodes.indices[node_inds]
                self.vtk_grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, nodes)

            self.vtk_grid.SetPoints(points)

        return self.vtk_grid


def vectors_to_np(input) -> np.ndarray:
    return np.reshape(input, (3, -1), order="F", copy=False)


def scalars_to_np(input) -> np.ndarray:
    return np.array(input, copy=False)
