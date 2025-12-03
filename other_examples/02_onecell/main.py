import os

import numpy as np

import tpysa
from opm.simulators import OnePhaseSimulator


def main(nx=1):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "mu": 3e9,  # 3.0 GPa
            "lambda": 2e9,  # 4.0 GPa
            "alpha": 1.0,  # O(1)
            "vtk_reset": False,
            "nx": nx,
            "n_time": 50,
            "n_total_cells": 2,
            # "vtk_writer": "Python",  # First run with "OPM", then "Python"
            "vtk_reset": False,
            "inv_spring_constant": 0,
        }
    )
    coupler = tpysa.Lagged

    # case_str = "ONECELL"
    case_str = "TWOCELLS_X"
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = tpysa.Biot_Model(
        opmcase,
        data,
        GridType=OnecellGrid,
        CouplerType=coupler,
        SimulatorType=OnePhaseSimulator,
    )
    model.simulate()


class OnecellGrid(tpysa.Grid):
    def tag_boundaries(self) -> None:
        num_cells_per_face = self.cell_faces.sum(axis=1)
        self.tags["domain_boundary_faces"] = num_cells_per_face != 0
        self.tags["domain_boundary_nodes"] = (
            self.face_nodes @ self.tags["domain_boundary_faces"]
        )

        # The default is to tag all boundaries as fixed
        self.tags["fixed_bdry"] = self.tags["domain_boundary_faces"].copy()
        self.tags["sprng_bdry"] = np.zeros_like(self.tags["domain_boundary_faces"])
        self.tags["free_bdry"] = np.zeros_like(self.tags["domain_boundary_faces"])


if __name__ == "__main__":
    main()
