import os
import sys

import numpy as np
from opm.simulators import OnePhaseSimulator

import tpysa


def main(coupler):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "inj_rate": 1e2,  # sm3/day
            "mu": 1e6,
            "n_time": 24,
            "n_total_cells": 20 * 15 * 9,
            "vtk_writer": "Python",  # First run with "OPM", then "Python"
            # "alpha": 0,
        }
    )

    case_str = "FAULT"
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = FaultBiot_Model(
        opmcase, data, FaultGrid, CouplerType=coupler, SimulatorType=OnePhaseSimulator
    )
    model.simulate()

    print("Done with coupler: {:}".format(coupler))


class FaultBiot_Model(tpysa.Biot_Model):
    """docstring for Biot_CartGrid."""

    def operate_wells(self, schedule):
        for well in schedule.get_wells(0):
            schedule.open_well(well.name, 0)

        for well in schedule.get_wells(12):
            schedule.shut_well(well.name, 12)


class FaultGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()
        self.tags["fixed_bdry"] = self.tags["domain_boundary_faces"]
        self.tags["free_bdry"] = np.zeros_like(self.tags["fixed_bdry"])

        # # Extract the "top" faces
        # bdry = self.tags["domain_boundary_faces"]

        # north = np.logical_and(bdry, self.face_centers[1] > 3490)
        # # east = np.logical_and(bdry, self.face_centers[0] > 3997)
        # south = np.logical_and(bdry, self.face_centers[1] < 2010)
        # # west = np.logical_and(bdry, self.face_centers[0] < 2002)

        # # Create a plane that lies in the middle of the domain
        # # origin = self.cell_centers[:, 1200]
        # # xpoint = self.cell_centers[:, 1219] - origin
        # # ypoint = self.cell_centers[:, 1480] - origin

        # # system = np.vstack((xpoint[:2], ypoint[:2]))
        # # rhs = np.array([xpoint[2], ypoint[2]])

        # # plane_grad = np.linalg.solve(system, rhs)
        # # plane_z = plane_grad @ (self.face_centers[:2, :] - origin[:2, None])
        # # plane_z += origin[2]

        # # top_half = self.face_centers[2, :] <= plane_z

        # self.tags["fixed_bdry"] = np.logical_or(south, north)

        # self.tags["sprng_bdry"] = np.logical_xor(
        #     self.tags["domain_boundary_faces"], self.tags["fixed_bdry"]
        # )


class Iterative_NP(tpysa.Iterative):
    global source_file

    def __init__(self, volumes, opmcase):
        super().__init__(volumes, opmcase)

        self.insource = np.load(source_file)["psi"]
        self.outsource = np.zeros_like(self.insource)

    def process_source(self, source, current_step=0, **kwargs) -> None:
        self.outsource[current_step] = source

    def get_source(self, current_step: int) -> np.ndarray:
        """
        Extracts the source for (t_i, t_{i + 1}] from the vtu-file at t_{i + 1}
        """
        return self.insource[current_step]

    def cleanup(self) -> None:
        out_file = list(os.path.splitext(source_file))
        out_file[0] = out_file[0] + "_out"
        new_source = "".join(out_file)

        np.savez(new_source, psi=self.outsource.reshape((-1, self.volumes.size)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
        coupler = Iterative_NP

    else:
        coupler = tpysa.Lagged

    main(coupler)
