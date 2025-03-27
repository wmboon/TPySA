import numpy as np
import os

import tpysa


def main(nx=10):
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
        "inj_rate": 0.05,  # sm3/day
        "nx": nx,
        "n_time": 50,
        "n_total_cells": nx**3,
    }
    coupler = tpysa.Lagged
    save_to_vtk = True

    ## Create a n x n x n Cartesian grid
    case_str = "GRID_" + str(nx)
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    model = CartBiot_Model(
        opmcase,
        data,
        save_to_vtk,
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

        # Clamp the remaining boundaries
        self.tags["displ_bdry"] = np.logical_xor(
            self.tags["domain_boundary_faces"], self.tags["tract_bdry"]
        )


if __name__ == "__main__":
    main()
