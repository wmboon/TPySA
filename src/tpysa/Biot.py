import numpy as np
from opm.simulators import BlackOilSimulator

from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig

import tpysa
import tpysa.simulator_reader


class Biot_Model:
    pass


def run_poromechanics(
    opmcase: str,
    data: dict,
    save_to_vtk: bool = False,
    CouplerType=tpysa.Lagged,
    GridType=tpysa.Grid,
    Simulator=BlackOilSimulator,
):
    ## Parse deck

    parser = Parser()
    data_file = "{}.DATA".format(opmcase)
    deck = parser.parse(data_file)

    ## Initialize flow simulator

    state = EclipseState(deck)
    schedule = Schedule(deck, state)

    # for well in schedule.get_wells(0):
    #     schedule.open_well(well.name, 0)

    # for well in schedule.get_wells(25):
    #     schedule.shut_well(well.name, 25)

    summary_config = SummaryConfig(deck, state, schedule)
    sim = Simulator(deck, state, schedule, summary_config)

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

        # assert np.allclose(sim.get_fluidstate_variable("Sw"), 1)
        var_dict = tpysa.simulator_reader.get_fluidstate_variables(sim)

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
        coupler.set_mass_source(grid, schedule, current_step, var_dict)

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
