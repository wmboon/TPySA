import os
from opm.simulators import GasWaterSimulator
import time
import numpy as np
import scipy.sparse.linalg as spla
import tpysa


def main():
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
        "n_total_cells": 46 * 73 * 31,
    }

    coupler = tpysa.Lagged
    save_to_vtk = True

    case_str = "DROGON"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    model = tpysa.Biot_Model(
        opmcase,
        data,
        save_to_vtk,
        SimulatorType=GasWaterSimulator,
        CouplerType=coupler,
        SolverType=HighTolSolver,
    )
    model.simulate()


class HighTolSolver(tpysa.AMGSolver):
    def solve(self, rhs: np.ndarray) -> tuple:
        start_time = time.time()

        num_it = 0

        def callback(_):
            nonlocal num_it
            num_it += 1
            print("BiCGStab: Iterate {:3}".format(num_it), end="\r")

        sol, info = spla.bicgstab(
            self.system,
            rhs,
            rtol=1e-3,
            M=self.precond,
            callback=callback,
        )
        self.report_time(
            "BiCGStab converged in {} iterations".format(num_it), start_time
        )

        return sol, info


if __name__ == "__main__":
    main()
