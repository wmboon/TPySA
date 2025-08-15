from opm.simulators import GasWaterSimulator

import tpysa


def main():
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "n_total_cells": 46 * 112 * 22,
            "vtk_writer": "Python",  # First run with "OPM", then "Python"
            "vtk_reset": False,
            "save_as_true": False,
            "compare_to_truth": True,
        }
    )
    coupler = tpysa.Lagged

    case_str = "NORNE"
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
