import unittest

import numpy as np
import os

import tpysa


class NNC_test(unittest.TestCase):
    def test_nnc_grid(self):
        case_str = "data/9_editnnc/9_EDITNNC"
        dir_name = os.path.dirname(__file__)
        opmcase = os.path.join(dir_name, case_str)

        egrid_file = f"{opmcase}.EGRID"
        grid = tpysa.Grid(egrid_file)

        # self.assertEqual(grid.num_cells, 8)

        self.assertTrue(
            np.allclose(grid.face_areas, np.linalg.norm(grid.face_normals, axis=0))
        )

        weight = np.sum(grid.face_centers * grid.face_normals, axis=0)
        cell_volumes = weight @ grid.cell_faces / grid.dim

        self.assertTrue(np.allclose(cell_volumes, grid.cell_volumes))

        pass


if __name__ == "__main__":
    unittest.main()
