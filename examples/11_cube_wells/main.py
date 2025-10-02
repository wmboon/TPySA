import os
import numpy as np
import sys
from opm.simulators import OnePhaseSimulator
import tpysa


def main(nx=10):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "perm": 1e3,
            "mu": 1e3,
            "lambda": 1e5,
            "nx": nx,
            "n_time": 24,
            "n_total_cells": nx**2,
            "vtk_writer": "Python",  # First run with "OPM", then "Python"
            "vtk_reset": False,
            # "alpha": 0,
        }
    )
    # coupler = Iterative
    coupler = tpysa.Lagged

    ## Create a n x n x n Cartesian grid
    case_str = "GRID_" + str(nx)
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = CartBiot_Model(
        opmcase, data, CartGrid, CouplerType=coupler, SimulatorType=OnePhaseSimulator
    )
    model.simulate()


class CartBiot_Model(tpysa.Biot_Model):
    """docstring for Biot_CartGrid."""

    def generate_deck(self):
        nx = self.data["nx"]
        self.data.update(
            {
                # "nx": nx,
                "ny": nx,
                "nz": 1,
                "hx": 100 / nx,
                "hy": 100 / nx,
                "hz": 100 / nx,
            }
        )

        dir_name = os.path.dirname(__file__)
        template_file = os.path.join(dir_name, "template/CARTGRID.DATA")
        tpysa.generate_deck_from_template(template_file, self.deck_file, self.data)

    # def operate_wells(self, schedule):
    #     for well in schedule.get_wells(0):
    #         schedule.open_well(well.name, 0)

    #     if len(schedule.reportsteps) > 20:
    #         for well in schedule.get_wells(20):
    #             schedule.shut_well(well.name, 20)


class CartGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()

        # Put zero traction on the top boundary
        self.tags["fixed_bdry"] = self.tags["domain_boundary_faces"].copy()
        # np.isclose(
        #     self.face_centers[2], np.min(self.face_centers[2])
        # )

        # Clamp the bottom boundary
        self.tags["free_bdry"] = np.zeros_like(self.tags["domain_boundary_faces"])
        # np.isclose(
        #     self.face_centers[2], np.max(self.face_centers[2])
        # )

        # Put springs on the remaining boundaries
        self.tags["sprng_bdry"] = np.zeros_like(self.tags["domain_boundary_faces"])
        # self.tags["domain_boundary_faces"].copy()
        # np.logical_xor(
        #     ,
        #     np.logical_or(self.tags["free_bdry"], self.tags["fixed_bdry"]),
        # )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_file = sys.argv[1]

    else:
        source_file = os.path.join(
            os.path.dirname(__file__), "fixed_point", "source_0.npz"
        )

    main()
