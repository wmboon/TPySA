import os
from opm.simulators import GasWaterSimulator
import tpysa


def main():
    ## Input: Model and discretization parameters
    data = {
        "mu": 3.5e9,  # 3.5 GPa
        "lambda": 4e9,  # 4.0 GPa
        "alpha": 0.87,  # O(1)
        "n_total_cells": 181 * 317 * 5,
    }

    coupler = tpysa.Lagged
    save_to_vtk = True

    case_str = "TROLL_COARSE_CLOSED"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    model = tpysa.Biot_Model(
        opmcase,
        data,
        save_to_vtk,
        SimulatorType=GasWaterSimulator,
        CouplerType=coupler,
        SolverType=tpysa.AMGSolver,
    )
    model.simulate()


if __name__ == "__main__":
    main()
