import numpy as np
import os

import tpysa


def main(nx=10):
    ## Input: Model and discretization parameters
    data = {
        "mu": 3.5e9,  # 3.5 GPa
        "lambda": 4e9,  # 4.0 GPa
        "alpha": 0.87,  # O(1)
        "inj_rate": 0.05,  # sm3/day
        "nx": nx,
        "n_time": 50,
        "n_total_cells": nx**3,
        "vtk_writer": "Python",  # First run with "OPM", then "Python"
        "vtk_reset": False,
    }
    coupler = tpysa.Iterative

    ## Create a n x n x n Cartesian grid
    case_str = "GRID_" + str(nx)
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    model = CartBiot_Model(
        opmcase,
        data,
        CartGrid,
        CouplerType=coupler,
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
        template_file = os.path.join(dir_name, "template/CARTGRID.DATA")
        tpysa.generate_deck_from_template(template_file, self.deck_file, self.data)

    def operate_wells(self, schedule):
        for well in schedule.get_wells(0):
            schedule.open_well(well.name, 0)

        if len(schedule.reportsteps) > 25:
            for well in schedule.get_wells(25):
                schedule.shut_well(well.name, 25)


class CartGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()

        # Put zero traction on the top boundary
        self.tags["tract_bdry"] = np.isclose(
            self.face_centers[2], np.min(self.face_centers[2])
        )

        # Clamp the bottom boundary
        self.tags["displ_bdry"] = np.isclose(
            self.face_centers[2], np.max(self.face_centers[2])
        )

        # Put springs on the remaining boundaries
        self.tags["sprng_bdry"] = np.logical_xor(
            self.tags["domain_boundary_faces"],
            np.logical_or(self.tags["tract_bdry"], self.tags["displ_bdry"]),
        )


if __name__ == "__main__":
    main()
