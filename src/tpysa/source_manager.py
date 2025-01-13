import numpy as np
from opm.io.ecl import EGrid
from opm.io.schedule import Schedule


def set_mass_source(grid: EGrid, schedule: Schedule, source: np.ndarray):

    # Make into array if it is a scalar
    if isinstance(source, np.ScalarType):
        source = np.full(grid.num_cells, source)

    # Convert to string
    source_str = source_stringer(grid, source)

    # Update the keyword in the schedule
    schedule.insert_keywords(source_str)


def source_stringer(grid: EGrid, source: np.ndarray):

    output = [""] * len(source)
    for c, s in enumerate(source):
        ijk = [i + 1 for i in grid.ijk_from_active_index(c)]
        output[c] = "\t {} {} {} WATER {} /".format(*ijk, s)

    return "\n".join(["SOURCE", *output, "/"])
