import os
import numpy as np

from opm.simulators import BlackOilSimulator

from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig

import tpysa

if __name__ == "__main__":

    ## Input: Model and discretization parameters
    data = {
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,  # O(1)
    }
    inj_rate = 0.05  # sm3/day
    nx = 10
    lagged = True

    ## Data management

    for key, item in data.items():
        data[key] = np.full(nx**3, item)  # Ensure the data entries are cell-wise

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

    ## Parse deck

    parser = Parser()
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
    fluid_p0 = sim.get_primary_variable("pressure")
    data["ref_pressure"] = fluid_p0.copy()

    ## Extract grid

    egrid_file = "{}.EGRID".format(opmcase)
    grid = tpysa.Grid(egrid_file)

    # Double check that ROCKBIOT is inserted appropriately
    field_props = state.field_props()
    rock_biot_ecl = field_props["ROCKBIOT"]
    assert np.allclose(data["rock_biot"], rock_biot_ecl)

    ## Initialize Mechanics

    tpsa_disc = tpysa.TPSA(grid)
    tpsa_disc.discretize(data)
    solid_p0 = np.zeros_like(fluid_p0)

    ## Choose coupling scheme
    n_time = len(schedule.reportsteps)
    n_space = grid.num_cells

    if lagged:
        coupler = tpysa.Lagged(n_space)
    else:
        coupler = tpysa.Iterative(n_space, n_time, opmcase)

    ## Ready to simulate
    reportsteps = schedule.reportsteps

    while not sim.check_simulation_finished():
        current_step = sim.current_step()
        dt = (reportsteps[current_step] - reportsteps[current_step - 1]).total_seconds()

        assert np.allclose(sim.get_fluidstate_variable("Sw"), 1)

        # Compute current fluid and solid pressures
        fluid_p = sim.get_primary_variable("pressure")
        displ, rotat, solid_p = tpsa_disc.solve(data, fluid_p)

        # Compute the changes in solid and fluid pressures
        # during the previous time step
        delta_fp = (fluid_p - fluid_p0) / dt
        delta_sp = (solid_p - solid_p0) / dt

        # Compute the mass source
        vol_source = -data["alpha"] / data["lambda"] * delta_sp

        # Set the mass source
        coupler.save_source(current_step, vol_source)
        coupler.set_mass_source(grid, schedule, current_step)

        # Save the pressures for the next time step
        fluid_p0 = fluid_p.copy()
        solid_p0 = solid_p.copy()

        if current_step == 10:
            vol_change = tpsa_disc.recover_volumetric_change(solid_p, fluid_p, data)
            tpysa.write_vtk(
                grid,
                [fluid_p, solid_p, displ, rotat, vol_change],
                ["fluid_p", "solid_p", "displacement", "rotation", "vol_change"],
                "{}_solutions.vtu".format(opmcase),
            )

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
