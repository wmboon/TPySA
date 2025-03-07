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
        "mu": 1e10,  # 10 GPa
        "lambda": 1e10,  # 10 GPa
        "alpha": 1,
        "gravity": -9.81 * 997,  # N / m3
    }
    lagged = False
    nx = 8

    inj_rate = 50  # kg/day

    rock_biot = data["alpha"] * data["alpha"] / data["lambda"]

    ## Create a n x n x n Cartesian grid
    case_str = "cartgrid/GRID_" + str(nx)
    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)
    data_file = f"{opmcase}.DATA"

    tpysa.generate_cart_grid(
        nx, output_file=data_file, rockbiot=rock_biot, time_steps=40
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

    if lagged:
        coupler = tpysa.Lagged(n_space)
    else:
        coupler = tpysa.Iterative(n_space, n_time, opmcase)

    ## Initial conditions
    fluid_p0 = sim.get_primary_variable("pressure")
    _, _, solid_p0 = tpsa_disc.solve(data, fluid_p0)
    dt = 1  # Doesn't matter because it will get overwritten

    reportsteps = schedule.reportsteps

    cc_local = grid.cell_centers - np.array([[0, 0, 1000]]).T
    inj_sites = np.max(cc_local, axis=0) <= 1 / 8 * 100
    pro_sites = np.min(cc_local, axis=0) >= 7 / 8 * 100

    inj_volume = np.sum(grid.cell_volumes[inj_sites])
    pro_volume = np.sum(grid.cell_volumes[pro_sites])

    while not sim.check_simulation_finished():
        current_step = sim.current_step()
        current_time = (reportsteps[current_step] - reportsteps[0]).days

        assert np.allclose(sim.get_fluidstate_variable("Sw"), 1)

        inj_source = np.zeros(grid.num_cells)
        if current_time >= 50 and current_time <= 250:
            inj_source[inj_sites] = inj_rate * grid.cell_volumes[inj_sites] / inj_volume
            inj_source[pro_sites] = (
                -inj_rate * grid.cell_volumes[pro_sites] / pro_volume
            )

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

        if current_step == 10:
            vol_change = tpsa_disc.recover_volumetric_change(solid_p, fluid_p, data)
            tpysa.write_vtk(
                grid,
                [fluid_p, solid_p, displ, rotat, vol_change],
                ["fluid_p", "solid_p", "displacement", "rotation", "vol_change"],
                "solutions.vtu",
            )

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
