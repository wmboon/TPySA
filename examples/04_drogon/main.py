import os
from opm.simulators import GasWaterSimulator
import tpysa


def main():
    ## Input: Model and discretization parameters
    data = {
        "mu": 3.5e9,  # 3.5 GPa
        "lambda": 4e9,  # 4.0 GPa
        "alpha": 0.87,  # O(1)
        "n_total_cells": 46 * 73 * 31,
        "rtol": 1e-4,  # Relative residual tolerance for the iterative solver
        "vtk_writer": "Python",  # First run with "OPM", then "Python"
    }

    coupler = tpysa.Lagged

    case_str = "DROGON"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    model = tpysa.Biot_Model(
        opmcase,
        data,
        SimulatorType=GasWaterSimulator,
        CouplerType=coupler,
    )
    model.simulate()


if __name__ == "__main__":
    main()
