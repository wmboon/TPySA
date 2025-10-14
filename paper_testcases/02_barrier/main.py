import os
import sys

import numpy as np

import tpysa


def main(coupler, source_file):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "inj_rate": 1e2,  # sm3/day
            "mu": 1e6,
            "n_time": 24,
            "n_total_cells": 20 * 15 * 9,
            # "vtk_writer": "Python",
            # "alpha": 0,
        }
    )

    case_str = "FAULT"
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = FaultBiot_Model(
        opmcase,
        data,
        FaultGrid,
        CouplerType=coupler,
        SolverType=tpysa.AMGSolver,
        mass_source_file=source_file,
    )
    model.simulate()

    print("Done with coupler: {:}".format(coupler))


class FaultBiot_Model(tpysa.Biot_Model):
    def operate_wells(self, schedule):
        for well in schedule.get_wells(0):
            schedule.open_well(well.name, 0)

        for well in schedule.get_wells(12):
            schedule.shut_well(well.name, 12)


class FaultGrid(tpysa.Grid):
    def tag_boundaries(self):
        super().tag_boundaries()
        self.tags["fixed_bdry"] = self.tags["domain_boundary_faces"]
        self.tags["free_bdry"] = np.zeros_like(self.tags["fixed_bdry"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_file = sys.argv[1]

    else:
        source_file = "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/anderson/source_true.npz"

    coupler = tpysa.Iterative

    main(coupler, source_file)
