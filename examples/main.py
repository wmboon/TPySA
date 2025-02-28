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

    data = {
        "mu": 1e9,  # 1 GPa
        "lambda": 1e9,  # 1 GPa
        "alpha": 1,
        "gravity": 0,
    }

    inj_rate = 100  # kg/day

    rock_biot = data["alpha"] * data["alpha"] / data["lambda"]

    # Create a n x n x n Cartesian grid
    case_str = "cartgrid/GRID_8"
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = f"{opmcase}.DATA"

    tpysa.generate_cart_grid(
        8, output_file=data_file, rockbiot=rock_biot, time_steps=30
    )

    ## Parse deck

    parser = Parser()
    deck = parser.parse(data_file)

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

    for key, item in data.items():
        data[key] = np.full(grid.num_cells, item)

    tpsa_disc.discretize(data)

    # double check that ROCKBIOT is inserted appropriately
    rock_biot = data["alpha"] * data["alpha"] / data["lambda"]

    field_props = state.field_props()
    rock_biot_ecl = field_props["ROCKBIOT"]

    assert np.allclose(rock_biot, rock_biot_ecl)

    ## Choose coupling scheme

    n_time = len(schedule.reportsteps)
    n_space = grid.num_cells

    # coupler = tpysa.Iterative(n_space, n_time, opmcase)
    coupler = tpysa.Lagged(n_space)

    ## Initial conditions
    fluid_p0 = sim.get_primary_variable("pressure")
    _, _, solid_p0 = tpsa_disc.solve(data, fluid_p0)
    dt = 1  # Doesn't matter because it will get overwritten

    reportsteps = schedule.reportsteps

    while not sim.check_simulation_finished():
        current_step = sim.current_step()

        assert np.allclose(sim.get_fluidstate_variable("Sw"), 1)

        inj_source = np.zeros(grid.num_cells)
        if current_step >= 5 and current_step <= 15:
            inj_source[0] = inj_rate
            inj_source[-1] = -inj_rate

        # Compute current fluid and solid pressures
        fluid_p = sim.get_primary_variable("pressure")
        displ, rotat, solid_p = tpsa_disc.solve(data, fluid_p)

        # Compute the changes in solid and fluid pressures
        # in the previous time step
        delta_fp = (fluid_p - fluid_p0) / dt
        delta_sp = (solid_p - solid_p0) / dt

        # Compute the mass source
        vol_source = -data["alpha"] / data["lambda"] * delta_sp

        # Set the mass source
        coupler.save_source(current_step, vol_source)
        coupler.set_mass_source(grid, schedule, current_step, inj_source)

        # Save the pressures for the next time step
        fluid_p0 = fluid_p.copy()
        solid_p0 = solid_p.copy()
        dt = (reportsteps[current_step + 1] - reportsteps[current_step]).total_seconds()

        # Advance
        sim.step()

    ## Postprocessing
    coupler.cleanup()
    sim.step_cleanup()

    ecl_summary = ESmry(f"{opmcase}.SMSPEC")
    BPR = ecl_summary["BPR:1,1,1"]
    time = ecl_summary["TIME"]

    ## Save the solution as numpy arrays
    array_str = "_".join((opmcase, str(data["alpha"][0]), coupler.str))
    np.savez(array_str, pressure=BPR, time=time)

    pass
