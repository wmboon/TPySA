import numpy as np
import os

import tpysa


def main():
    ## Input: Model and discretization parameters
    data = {
        "mu": 3.5e9,  # 3.5 GPa
        "lambda": 4e9,  # 4.0 GPa
        "alpha": 0.87,  # O(1)
        "inj_rate": 1e3,  # sm3/day
        "n_time": 30,
        "n_total_cells": 20 * 15 * 9,
        "vtk_writer": "Python",  # First run with "OPM", then "Python"
    }
    coupler = tpysa.Lagged

    case_str = "FAULT"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    model = FaultBiot_Model(opmcase, data, FaultGrid, CouplerType=coupler)
    model.simulate()


class FaultBiot_Model(tpysa.Biot_Model):
    """docstring for Biot_CartGrid."""

    def operate_wells(self, schedule):
        for well in schedule.get_wells(0):
            schedule.open_well(well.name, 0)

        for well in schedule.get_wells(25):
            schedule.shut_well(well.name, 25)


class FaultGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()

        # Extract the "top" faces
        bdry = self.tags["domain_boundary_faces"]

        north = np.logical_and(bdry, self.face_centers[1] > 3490)
        # east = np.logical_and(bdry, self.face_centers[0] > 3997)
        south = np.logical_and(bdry, self.face_centers[1] < 2010)
        # west = np.logical_and(bdry, self.face_centers[0] < 2002)

        # Create a plane that lies in the middle of the domain
        # origin = self.cell_centers[:, 1200]
        # xpoint = self.cell_centers[:, 1219] - origin
        # ypoint = self.cell_centers[:, 1480] - origin

        # system = np.vstack((xpoint[:2], ypoint[:2]))
        # rhs = np.array([xpoint[2], ypoint[2]])

        # plane_grad = np.linalg.solve(system, rhs)
        # plane_z = plane_grad @ (self.face_centers[:2, :] - origin[:2, None]) + origin[2]

        # top_half = self.face_centers[2, :] <= plane_z

        self.tags["displ_bdry"] = np.logical_or(south, north)

        self.tags["sprng_bdry"] = np.logical_xor(
            self.tags["domain_boundary_faces"], self.tags["tract_bdry"]
        )


if __name__ == "__main__":
    main()
