import os
import numpy as np

from opm.simulators import BlackOilSimulator

from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig

import tpysa


def run_poromechanics(
    opmcase: str,
    data: dict,
    save_to_vtk: bool = False,
    CouplerType=tpysa.Lagged,
    GridType=tpysa.Grid,
):
    ## Parse deck

    parser = Parser()
    data_file = "{}.DATA".format(opmcase)
    deck = parser.parse(data_file)

    ## Initialize flow simulator

    state = EclipseState(deck)
    schedule = Schedule(deck, state)

    schedule.open_well("INJE", 5)
    schedule.open_well("PROD", 5)

    schedule.shut_well("INJE", 25)
    schedule.shut_well("PROD", 25)

    summary_config = SummaryConfig(deck, state, schedule)
    sim = BlackOilSimulator(deck, state, schedule, summary_config)

    ## Initial conditions
    sim.step_init()  # Creates the EGRID file
    fluid_p = sim.get_primary_variable("pressure")
    data["ref_pressure"] = fluid_p.copy()

    ## Extract grid
    egrid_file = "{}.EGRID".format(opmcase)
    grid = GridType(egrid_file)

    # Double check that ROCKBIOT is inserted appropriately
    field_props = state.field_props()
    rock_biot_ecl = field_props["ROCKBIOT"]
    assert np.allclose(data["rock_biot"], rock_biot_ecl)

    ## Initialize Mechanics

    tpsa_disc = tpysa.TPSA(grid)
    tpsa_disc.discretize(data)
    solid_p0 = np.zeros_like(fluid_p)

    ## Choose coupling scheme
    n_time = len(schedule.reportsteps)
    n_space = grid.num_cells

    coupler = CouplerType(n_space, n_time, opmcase)

    ## Ready to simulate
    reportsteps = schedule.reportsteps

    while not sim.check_simulation_finished():
        current_step = sim.current_step()
        dt = (reportsteps[current_step] - reportsteps[current_step - 1]).total_seconds()

        assert np.allclose(sim.get_fluidstate_variable("Sw"), 1)

        # Compute current fluid and solid pressures
        fluid_p = sim.get_primary_variable("pressure")
        displ, rotat, solid_p = tpsa_disc.solve(data, fluid_p)

        # Compute the change in solid pressures
        # during the previous time step
        delta_sp = (solid_p - solid_p0) / dt

        # Compute the mass source
        vol_source = -data["alpha"] / data["lambda"] * delta_sp

        # Set the mass source
        coupler.save_source(current_step, vol_source)
        coupler.set_mass_source(grid, schedule, current_step)

        # Save the solid pressure for the next time step
        solid_p0 = solid_p.copy()

        if save_to_vtk:
            vol_change = tpsa_disc.recover_volumetric_change(solid_p, fluid_p, data)
            sol_dict = {
                "pressure_fluid": fluid_p,
                "pressure_solid": solid_p,
                "displacement": displ,
                "rotation": rotat,
                "vol_change": vol_change,
            }
            tpysa.write_vtk(grid, sol_dict, opmcase, current_step)

        # Advance
        sim.step()

    ## Postprocessing
    coupler.cleanup()
    sim.step_cleanup()

    # ecl_summary = ESmry("{}.SMSPEC".format(opmcase))
    # BPR = ecl_summary["BPR:1,1,1"]
    # time = ecl_summary["TIME"]

    ## Save the solution as numpy arrays
    # array_str = "_".join((opmcase, str(data["alpha"][0]), coupler.str))
    # np.savez(array_str, pressure=BPR, time=time)

    pass


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

    tpysa.generate_cart_grid(
        nx,
        output_file=data_file,
        rockbiot=data["rock_biot"],
        inj_rate=inj_rate,
        time_steps=40,
    )

    run_poromechanics(opmcase, data, save_to_vtk, coupler, tpysa.CartGrid)


def faulted_grid_example():
    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
    }
    inj_rate = 0.05  # sm3/day
    coupler = tpysa.Lagged
    save_to_vtk = True

    num_cells = 20 * 15 * 9

    ## Data management

    for key, item in data.items():
        data[key] = np.full(num_cells, item)  # Ensure the data entries are cell-wise

    data["rock_biot"] = data["alpha"] * data["alpha"] / data["lambda"]  # 1/Pa

    ## Create a n x n x n Cartesian grid

    case_str = "fault_grid/FAULT"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = "{}.DATA".format(opmcase)

    tpysa.generate_faulted_grid(
        output_file=data_file,
        rockbiot=data["rock_biot"],
        inj_rate=inj_rate,
        time_steps=40,
    )

    run_poromechanics(opmcase, data, save_to_vtk, coupler)


if __name__ == "__main__":
    cartgrid_example(nx=10)
    # faulted_grid_example()
