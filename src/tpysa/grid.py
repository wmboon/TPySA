import numpy as np
import scipy.sparse as sps
import vtk
from opm.io.ecl import EclFile, EGrid
from opmcpg._cpggrid import (
    UnstructuredGrid,
    index_vector,
    process_cpg_grid,
    value_vector,
)
from vtkmodules.util.numpy_support import numpy_to_vtk


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
        self.check_boundary_tags()

        self.actnum = self.actnum.astype(bool)

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

        self.tags["sprng_bdry"] = self.tags["domain_boundary_faces"].copy()
        self.tags["displ_bdry"] = np.zeros_like(self.tags["domain_boundary_faces"])

    def check_boundary_tags(self):
        # Include the traction boundaries in the spring boundary tag
        if "tract_bdry" in self.tags:
            self.tags["sprng_bdry"] = np.logical_or(
                self.tags["sprng_bdry"], self.tags["tract_bdry"]
            )

        # Check if all boundary faces are tagged exactly once
        coverage_check = np.logical_or(self.tags["sprng_bdry"], self.tags["displ_bdry"])
        assert np.all(coverage_check == self.tags["domain_boundary_faces"])
        assert ~np.any(np.logical_and(self.tags["sprng_bdry"], self.tags["displ_bdry"]))

    def get_vtk(self) -> vtk.vtkUnstructuredGrid:
        if not hasattr(self, "vtk_grid"):
            cell_nodes = self.compute_cell_nodes()

            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(self.nodes.T))

            faces = vtk.vtkCellArray()
            faceLocations = vtk.vtkCellArray()

            cells = vtk.vtkCellArray()
            cellTypes = vtk.vtkUnsignedCharArray()

            for cell in np.arange(self.num_cells):
                cellTypes.InsertNextValue(vtk.VTK_HEXAHEDRON)

                loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
                node_inds = cell_nodes.indices[loc]

                if len(node_inds) != 8:
                    xyz = np.vstack(self.xyz_from_active_index(cell))
                    keep_node = np.empty(len(node_inds), dtype=bool)

                    for ind, node in enumerate(node_inds):
                        dist = np.linalg.norm(
                            xyz - self.nodes[:, node][:, None], ord=np.inf, axis=0
                        )
                        keep_node[ind] = np.min(dist) <= 1e-2

                    assert keep_node.sum() == 8

                    node_inds = node_inds[keep_node]
                cells.InsertNextCell(8, node_inds)

            self.vtk_grid = vtk.vtkUnstructuredGrid()
            self.vtk_grid.SetPoints(points)
            self.vtk_grid.SetPolyhedralCells(cellTypes, cells, faceLocations, faces)

        return self.vtk_grid


def vectors_to_np(input) -> np.ndarray:
    return np.reshape(input, (3, -1), order="F", copy=False)


def scalars_to_np(input) -> np.ndarray:
    return np.array(input, copy=False)
