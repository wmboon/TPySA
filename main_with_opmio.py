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

## Input

dir_name = os.path.dirname(__file__)
opmcase = os.path.join(dir_name, "data_single_phase/SINGLE_PHASE")

data_file = f"{opmcase}.DATA"
deck = Parser().parse(data_file)

## Initialize flow simulator

state = EclipseState(deck)
schedule = Schedule(deck, state)
summary_config = SummaryConfig(deck, state, schedule)

sim = BlackOilSimulator(deck, state, schedule, summary_config)
sim.step_init()  # This creates the EGRID file

## Extract grid

egrid_file = f"{opmcase}.EGRID"
grid = CartEGrid(egrid_file)

## Initialize Mechanics

tpsa = TPSA(grid)
data = {
    "mu": np.full(grid.num_cells, 1),
    "lambda": np.full(grid.num_cells, 1),
    "l2": np.full(grid.num_cells, 1),
    "alpha": np.full(grid.num_cells, 1),
    "gravity": np.full(grid.num_cells, 0)
}
tpsa.discretize(data)

## Simulate

p0 = sim.get_primary_variable("pressure")
sp0 = np.zeros_like(p0)

sim.step()  # do one time step so that we get access to get_dt()

while not sim.check_simulation_finished():

    """ # current_step = sim.current_step()
    # if current_step == 5:
    #     set_mass_source(grid, schedule, 1e3)
    # elif current_step == 15:
    #     set_mass_source(grid, schedule, 0) """

    # Compute new water and solid pressures
    p = sim.get_primary_variable("pressure")
    u, r, sp = tpsa.solve(data, p)

    # Compute the changes in pressures
    dt = sim.get_dt()
    delta_p = (p - p0) / dt
    delta_sp = (sp - sp0) / dt

    # Set the new mass source
    source = delta_sp + data["alpha"] * delta_p
    source *= -data["alpha"] / data["lambda"]

    set_mass_source(grid, schedule, source)

    # Save the pressures for the next time step
    p0 = p.copy()
    sp0 = sp.copy()

    # Advance
    sim.step()

sim.step_cleanup()

ecl_summary = ESmry(f"{opmcase}.SMSPEC")

## Postprocessing
time = ecl_summary["TIME"]
BPR = ecl_summary["BPR:1,1,1"]

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(time, BPR)
# plt.show()

ax = plt.figure(1).add_subplot(projection="3d")
x, y, z = grid.cell_centers
u_x, u_y, u_z = u.reshape((3, -1), order="F") / np.amax(u)
ax.quiver(x, y, z, u_x, u_y, u_z)

plt.show()

pass
## Random handy tools
# grid_file = EclFile(f"{opmcase}.EGRID")
# init_file = EclFile(f"{opmcase}.INIT")

# nnc1 = grid_file["NNC1"]
# nnc2 = grid_file["NNC2"]
# tran = init_file["TRANNNC"]

# nnc_list = []
# for g1, g2, t in zip(nnc1, nnc2, tran):
#     nnc_list.append((g1, g2, t))


# grid = state.grid()
