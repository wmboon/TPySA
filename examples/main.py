import os
import numpy as np

from opm.simulators import GasWaterSimulator
import tpysa
import tpysa.templates


def norne_example():
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
    }

    coupler = tpysa.Lagged
    save_to_vtk = True
    num_cells = 44431

    for key, item in data.items():
        data[key] = np.full(num_cells, item)  # Ensure the data entries are cell-wise

    data["rock_biot"] = data["alpha"] * data["alpha"] / data["lambda"]  # 1/Pa

    case_str = "norne/NORNE"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = "{}.DATA".format(opmcase)

    tpysa.templates.generate_norne_deck(data["rock_biot"], data_file)
    tpysa.run_poromechanics(
        opmcase, data, save_to_vtk, coupler, Simulator=GasWaterSimulator
    )


def drogon_example():
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
    }

    coupler = tpysa.Lagged
    save_to_vtk = True
    num_cells = 70972

    for key, item in data.items():
        data[key] = np.full(num_cells, item)  # Ensure the data entries are cell-wise

    data["rock_biot"] = data["alpha"] * data["alpha"] / data["lambda"]  # 1/Pa

    case_str = "drogon/DROGON"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = "{}.DATA".format(opmcase)

    tpysa.templates.generate_drogon_deck(data["rock_biot"], data_file)
    tpysa.run_poromechanics(
        opmcase, data, save_to_vtk, coupler, Simulator=GasWaterSimulator
    )


def cartgrid_example(nx=5):
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
    }
    inj_rate = 0.05  # sm3/day
    coupler = tpysa.Lagged
    save_to_vtk = True

    num_cells = nx**3

    ## Data management

    for key, item in data.items():
        data[key] = np.full(num_cells, item)  # Ensure the data entries are cell-wise

    data["rock_biot"] = data["alpha"] * data["alpha"] / data["lambda"]  # 1/Pa

    ## Create a n x n x n Cartesian grid

    case_str = "cartgrid/GRID_" + str(nx)
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = "{}.DATA".format(opmcase)

    tpysa.generate_cart_deck(
        nx,
        output_file=data_file,
        rockbiot=data["rock_biot"],
        inj_rate=inj_rate,
        time_steps=40,
    )

    tpysa.run_poromechanics(opmcase, data, save_to_vtk, coupler, tpysa.CartGrid)


def faulted_grid_example():
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e12,  # 100 GPa
        "alpha": 1,  # O(1)
    }
    inj_rate = 1e3  # sm3/day
    coupler = tpysa.Lagged
    save_to_vtk = True

    num_cells = 20 * 15 * 9

    ## Data management

    for key, item in data.items():
        data[key] = np.full(num_cells, item)  # Ensure the data entries are cell-wise

    data["rock_biot"] = data["alpha"] * data["alpha"] / data["lambda"]  # 1/Pa

    ## Create the faulted grid from template

    case_str = "fault_grid/FAULT"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = "{}.DATA".format(opmcase)

    tpysa.generate_faulted_deck(
        output_file=data_file,
        rockbiot=data["rock_biot"],
        inj_rate=inj_rate,
        time_steps=50,
    )

    tpysa.run_poromechanics(opmcase, data, save_to_vtk, coupler, tpysa.FaultGrid)


if __name__ == "__main__":
    # cartgrid_example(nx=10)
    # faulted_grid_example()
    # norne_example()
    drogon_example()
