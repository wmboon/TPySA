import os
import numpy as np

from opm.simulators import BlackOilSimulator

from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.io.ecl import EGrid, ESmry, EclFile

from src.source_stringer import source_stringer
from src.cartgrid import CartGrid

## Input

dir_name = os.path.dirname(__file__)
opmcase = os.path.join(dir_name, "data_single_phase/SINGLE_PHASE")

data_file = f"{opmcase}.DATA"
parser = Parser()
deck = parser.parse(data_file)

## Initialize Simulator

state = EclipseState(deck)
schedule = Schedule(deck, state)
summary_config = SummaryConfig(deck, state, schedule)

sim = BlackOilSimulator(deck, state, schedule, summary_config)
sim.step_init()

# grid_file = EclFile(f"{opmcase}.EGRID")
# init_file = EclFile(f"{opmcase}.INIT")

# nnc1 = grid_file["NNC1"]
# nnc2 = grid_file["NNC2"]
# tran = init_file["TRANNNC"]

# nnc_list = []
# for g1, g2, t in zip(nnc1, nnc2, tran):
#     nnc_list.append((g1, g2, t))


# grid = state.grid()
egrid = EGrid(f"{opmcase}.EGRID")
cgrid = CartGrid(egrid)

## Simulate

p = sim.get_primary_variable("pressure")

while not sim.check_simulation_finished():
    if sim.current_step() == 5:
        source = np.full(egrid.active_cells, 1e3)

        source_str = source_stringer(egrid, source)
        schedule.insert_keywords(source_str)

    elif sim.current_step() == 15:
        source = np.full(egrid.active_cells, 0)

        source_str = source_stringer(egrid, source)
        schedule.insert_keywords(source_str)
    sim.step()

sim.step_cleanup()

ecl_summary = ESmry(f"{opmcase}.SMSPEC")

## Postprocessing

time = ecl_summary["TIME"]
BPR = ecl_summary["BPR:1,1,1"]

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(time, BPR)
plt.show()

pass
