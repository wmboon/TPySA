import numpy as np


def source_stringer(grid, source: np.ndarray):

    output = [""] * len(source)
    for c, s in enumerate(source):
        ijk = [i + 1 for i in grid.ijk_from_active_index(c)]
        output[c] = "\t {} {} {} WATER {} /".format(*ijk, s)

    return "\n".join(["SOURCE", *output, "/"])
