import os

import numpy as np

import tpysa
from opm.simulators import OnePhaseSimulator


def main(nx=5):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "inj_rate": 0.05,  # sm3/day
            "nx": nx,
            "n_time": 20,
            # "mu": 3e9,
            # "alpha": 1.0,
            # "lambda": 2e9,
            "n_total_cells": nx**3,
            "vtk_writer": "Python",  # First run with "OPM", then "Python"
            "vtk_reset": False,
        }
    )
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
                "nz": nx,
                "hx": 100 / nx,
                "hy": 100 / nx,
                "hz": 100 / nx,
            }
        )

        dir_name = os.path.dirname(__file__)
        template_file = os.path.join(dir_name, "template/GRID_5_BC_FREE.DATA")
        tpysa.generate_deck_from_template(template_file, self.deck_file, self.data)

    def operate_wells(self, schedule):
        for well in schedule.get_wells(0):
            schedule.open_well(well.name, 0)

        # if len(schedule.reportsteps) > 25:
        #     for well in schedule.get_wells(25):
        #         schedule.shut_well(well.name, 25)


class CartGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()

        # self.tags["fixed_bdry"] = self.tags["domain_boundary_faces"].copy()
        self.tags["fixed_bdry"] = np.logical_or(
            np.isclose(self.face_centers[2], np.min(self.face_centers[2])),
            np.isclose(self.face_centers[2], np.max(self.face_centers[2])),
        )
        self.tags["free_bdry"] = np.logical_xor(
            self.tags["domain_boundary_faces"], self.tags["fixed_bdry"]
        )
        self.tags["sprng_bdry"] = np.zeros_like(self.tags["free_bdry"])
        # pass
        # np.isclose(
        #     self.face_centers[2], np.min(self.face_centers[2])
        # )

        # Clamp the bottom boundary
        # np.zeros_like(self.tags["domain_boundary_faces"])

        # Put springs on the remaining boundaries
        # self.tags["sprng_bdry"] = np.logical_xor(
        #     self.tags["domain_boundary_faces"],
        #     np.logical_or(self.tags["free_bdry"], self.tags["fixed_bdry"]),
        # )


if __name__ == "__main__":
    main()
