import os
import numpy as np

from opm.simulators import BlackOilSimulator

from opm.io import Parser
from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.io.ecl import ESmry, EclFile

from src.source_manager import set_mass_source
from src.cartgrid import CartEGrid
from src.TPSA import TPSA


class Lagged:
    """docstring for Lagged coupler."""

    def __init__(self, n_time, n_space, opm_case=""):
        self.source = np.zeros(n_space)
        self.str = "lagged"

    def save_source(self, current_step, source):
        self.source = source

    def get_source(self, current_step):
        return self.source

    def cleanup(self):
        pass


class Iterative:
    """docstring for Iterative coupler."""

    def __init__(self, n_time, n_space, opm_case=""):
        self.sources_file = f"{opm_case}_sources.npz"
        self.str = "iterative"

        try:
            self.source = np.load(self.sources_file)["source"]
        except Exception:
            self.source = np.zeros((n_time, n_space))

        if self.source.shape != (n_time, n_space):
            self.source = np.zeros((n_time, n_space))

    def save_source(self, current_step, source):
        self.source[current_step - 1] = source

    def get_source(self, current_step):
        return self.source[current_step]

    def cleanup(self):
        np.savez(self.sources_file, source=self.source)


if __name__ == "__main__":

    ## Input

    case_str = "data_single_phase/SINGLE_PHASE"

    dir_name = os.path.dirname(__file__)
    opmcase = os.path.join(dir_name, case_str)

    data_file = f"{opmcase}.DATA"
    deck = Parser().parse(data_file)

    ## Initialize flow simulator

    state = EclipseState(deck)
    schedule = Schedule(deck, state)
    summary_config = SummaryConfig(deck, state, schedule)
    sim = BlackOilSimulator(deck, state, schedule, summary_config)

    sim.step_init()  # Create the EGRID file

    ## Extract grid

    egrid_file = f"{opmcase}.EGRID"
    grid = CartEGrid(egrid_file)

    ## Initialize Mechanics

    tpsa_disc = TPSA(grid)
    data = {
        "mu": np.full(grid.num_cells, 5e1),
        "lambda": np.full(grid.num_cells, 5e1),
        "l2": np.full(grid.num_cells, 1),
        "alpha": np.full(grid.num_cells, 1),
        "gravity": np.full(grid.num_cells, 0),
    }
    tpsa_disc.discretize(data)

    ## Choose coupling scheme

    n_time = len(schedule.timesteps)
    n_space = state.grid().nactive

    coupler = Iterative(n_time, n_space, opmcase)
    # coupler = Lagged(n_time, n_space)

    ## Simulate

    fluid_p0 = sim.get_primary_variable("pressure")
    solid_p0 = np.zeros_like(fluid_p0)

    set_mass_source(grid, schedule, coupler.get_source(0))
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
        set_mass_source(grid, schedule, coupler.get_source(current_step))

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
    array_str = opmcase + coupler.str
    np.savez(array_str, p=BPR, u=displ, p_s=solid_p, r_s=rotat, t=time)

    pass
