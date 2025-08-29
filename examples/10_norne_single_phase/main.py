import os
import numpy as np
import sys
from opm.simulators import OnePhaseSimulator
import tpysa


def main():
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "n_total_cells": 46 * 112 * 22,
            "vtk_writer": "Python",  # First run with "OPM", then "Python"
            "vtk_reset": False,
            "n_time_steps": 30,
            # "alpha": 0,
        }
    )
    # coupler = Iterative_NP
    coupler = tpysa.Lagged

    case_str = "NORNE"
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = Norne_Model(
        opmcase,
        data,
        GridType=NorneGrid,
        SimulatorType=OnePhaseSimulator,
        CouplerType=coupler,
    )
    model.simulate()


class Norne_Model(tpysa.Biot_Model):
    """docstring for Biot_CartGrid."""

    def operate_wells(self, schedule):
        for well in schedule.get_wells(0):
            schedule.open_well(well.name, 0)

        if len(schedule.reportsteps) > 30:
            half_time = 15
            for well in schedule.get_wells(half_time):
                schedule.shut_well(well.name, half_time)


class NorneGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()
        # self.tags["fixed_bdry"] = self.tags["domain_boundary_faces"].copy()


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
        return self.insource[current_step + 1]

    def cleanup(self) -> None:
        out_file = list(os.path.splitext(source_file))
        out_file[0] = out_file[0] + "_out"
        new_source = "".join(out_file)

        np.savez(new_source, psi=self.outsource.reshape((-1, self.volumes.size)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_file = sys.argv[1]

    else:
        source_file = os.path.join(
            os.path.dirname(__file__), "fixed_point", "source_0.npz"
        )

    main()
