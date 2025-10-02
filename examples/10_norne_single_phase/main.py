import os
import sys

import numpy as np

import tpysa


def main(coupler, source_file):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "n_total_cells": 46 * 112 * 22,
            "n_time_steps": 30,
            # "vtk_writer": "Python",
        }
    )

    case_str = "NORNE"
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = Norne_Model(
        opmcase,
        data,
        GridType=NorneGrid,
        CouplerType=coupler,
        mass_source_file=source_file,
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
        coupler = tpysa.Iterative

    else:
        source_file = "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/10_norne_single_phase/anderson/source_true.npz"
        coupler = tpysa.Lagged

    main(coupler, source_file)
