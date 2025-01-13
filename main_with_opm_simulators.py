import os
import numpy as np

from opm.simulators import BlackOilSimulator

from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.io.ecl import ESmry

import tpysa

if __name__ == "__main__":

    ## Parse deck

    # case_str = "tests/data/four_blocks_fullshift/FOURBLOCKS"
    case_str = "tests/data/single_phase/SINGLE_PHASE"
    # case_str = "tests/data/spe1/SPE1CASE1"

    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    data_file = f"{opmcase}.DATA"
    deck = Parser().parse(data_file)

    ## Initialize flow simulator

    state = EclipseState(deck)
    schedule = Schedule(deck, state)
    summary_config = SummaryConfig(deck, state, schedule)
    sim = BlackOilSimulator(deck, state, schedule, summary_config)

    sim.step_init()  # Creates the EGRID file

    ## Extract grid

    egrid_file = f"{opmcase}.EGRID"
    grid = tpysa.Grid(egrid_file)

    ## Initialize Mechanics

    tpsa_disc = tpysa.TPSA(grid)
    data = {
        "mu": np.full(grid.num_cells, 5e1),
        "lambda": np.full(grid.num_cells, 5e1),
        "l2": np.full(grid.num_cells, 1),
        "alpha": np.full(grid.num_cells, 1),
        "gravity": np.full(grid.num_cells, 0),
    }
    tpsa_disc.discretize(data)

    # double check that ROCKBIOT is inserted appropriately
    rock_biot = data["alpha"] * data["alpha"] / data["lambda"]
    rock_biot_ecl = state.field_props()["ROCKBIOT"]

    if not np.allclose(rock_biot, rock_biot_ecl):
        import warnings

        warnings.warn("Mismatch between the rock_biot coefficient input.")

    ## Choose coupling scheme

    n_time = len(schedule.reportsteps)
    n_space = grid.num_cells

    # coupler = tpysa.Iterative(n_space, n_time, opmcase)
    coupler = tpysa.Lagged(n_space)

    ## Simulate

    fluid_p0 = sim.get_primary_variable("pressure")
    solid_p0 = np.zeros_like(fluid_p0)

    coupler.set_mass_source(grid, schedule, 0)
    sim.step()  # do the first time step so that we get access to get_dt()

    while not sim.check_simulation_finished():
        dt = sim.get_dt()
        current_step = sim.current_step()

        if current_step == 8 or current_step == 16:
            injection_rate = 1e7 / dt
        else:
            injection_rate = 0

        # Compute new fluid and solid pressures
        fluid_p = sim.get_primary_variable("pressure")
        displ, rotat, solid_p = tpsa_disc.solve(data, fluid_p)

        # Compute the changes in pressures
        delta_fp = (fluid_p - fluid_p0) / dt
        delta_sp = (solid_p - solid_p0) / dt

        # Compute the mass source
        source = delta_sp + data["alpha"] * delta_fp
        source *= -data["alpha"] / data["lambda"]
        source += injection_rate

        # Set the mass source
        coupler.save_source(current_step, source)
        coupler.set_mass_source(grid, schedule, current_step)

        # Save the pressures for the next time step
        fluid_p0 = fluid_p.copy()
        solid_p0 = solid_p.copy()

        # Advance
        sim.step()

    ## Postprocessing
    coupler.cleanup()
    sim.step_cleanup()

    ecl_summary = ESmry(f"{opmcase}.SMSPEC")
    time = ecl_summary["TIME"]
    BPR = ecl_summary["BPR:5,5,5"]

    ## Save the solution as numpy arrays
    array_str = opmcase + "_" + coupler.str
    np.savez(array_str, p=BPR, u=displ, p_s=solid_p, r_s=rotat, t=time)

    pass
