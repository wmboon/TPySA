import numpy as np

from opm.io.ecl import EGrid
from opm.io.schedule import Schedule


class Coupler:
    def set_mass_source(self, grid: EGrid, schedule: Schedule, current_step: int):
        source = self.get_source(current_step)

        # Make into array if it is a scalar
        if isinstance(source, np.ScalarType):
            source = np.full(grid.num_cells, source)

        # Convert to string
        source_str = self.source_to_str(grid, source)

        # Update the keyword in the schedule
        schedule.insert_keywords(source_str)

    def source_to_str(self, grid: EGrid, source: np.ndarray):

        output = [""] * len(source)
        for c, s in enumerate(source):
            ijk = [i + 1 for i in grid.ijk_from_active_index(c)]
            output[c] = "\t {} {} {} WATER {} /".format(*ijk, s)

        return "\n".join(["SOURCE", *output, "/"])


class Lagged(Coupler):
    """docstring for Lagged coupler."""

    def __init__(self, n_space):
        self.source = np.zeros(n_space)
        self.str = "lagged"

    def save_source(self, current_step, source):
        self.source = source

    def get_source(self, current_step):
        return self.source

    def cleanup(self):
        pass


class Iterative(Coupler):
    """docstring for Iterative coupler."""

    def __init__(self, n_space, n_time, opm_case=""):
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
