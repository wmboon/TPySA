from opm.simulators import GasWaterSimulator
import tpysa


def main():
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "n_total_cells": 181 * 317 * 217,
            "rtol": 1e-3,  # Relative residual tolerance for the iterative solver
            "vtk_writer": "Python",  # First run with "OPM", then "Python"
        }
    )
    coupler = tpysa.Lagged

    case_str = "TROLL"
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = tpysa.Biot_Model(
        opmcase,
        data,
        SimulatorType=GasWaterSimulator,
        CouplerType=coupler,
    )
    model.simulate()


if __name__ == "__main__":
    main()
