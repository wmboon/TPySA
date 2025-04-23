import numpy as np
import logging
import tpysa
from opm.io.ecl import EGrid
from opm.io.schedule import Schedule


class Coupler:
    def set_mass_source(
        self, grid: EGrid, schedule: Schedule, current_step: int, variables: dict
    ):
        source = self.get_source(current_step)

        # Make into array if it is a scalar
        if isinstance(source, np.ScalarType):
            source = np.full(grid.num_cells, source)

        # Convert units from m3/s to kg/day
        source *= 24 * 60 * 60  # from 1/second to 1/day
        source *= variables["rho_w"]  # from m^3 to kg

        # Convert to string
        source_str = self.source_to_str(grid, source)

        # Update the keyword in the schedules
        schedule.insert_keywords(source_str)

    def source_to_str(self, grid: EGrid, source: np.ndarray):
        output = [""] * len(source)
        for c, s in enumerate(source):
            ijk = [i + 1 for i in grid.ijk_from_active_index(c)]
            output[c] = "\t {} {} {} WATER {} /".format(*ijk, s)

        return "\n".join(["SOURCE", *output, "/"])


class Lagged(Coupler):
    """docstring for Lagged coupler."""

    def __init__(self, n_space, *_):
        self.source = np.zeros(n_space)
        self.str = "lagged"

    def save_source(self, source, *_):
        self.source = source

    def get_source(self, *_):
        return self.source

    def cleanup(self):
        pass


class Iterative(Coupler):
    """docstring for Iterative coupler."""

    def __init__(self, n_space, opmcase: str):
        self.source = np.zeros(n_space)
        self.opmcase = opmcase
        self.str = "iterative"

        self.num_cells = n_space
        self.sqrd_diff_source = 0.0
        self.sqrd_norm_source = 0.0

        logger = logging.getLogger()
        ch = logging.FileHandler(self.opmcase + ".ITER")
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    def save_source(self, source, current_step):
        if current_step > 0:
            tpysa.write_vtk(
                {"vol_source": source}, self.opmcase, current_step - 1, self.num_cells
            )

        diff = source - self.source
        self.sqrd_diff_source += np.dot(diff, diff)
        self.sqrd_norm_source += np.dot(source, source)

    def get_source(self, current_step):
        self.source = tpysa.read_source_from_vtk(
            self.opmcase, current_step, self.num_cells
        )
        return self.source.astype(float, copy=True)

    def cleanup(self):
        logging.error(
            "Source difference, abs: {:.2e}, rel: {:.4e}".format(
                np.sqrt(self.sqrd_diff_source),
                np.sqrt(self.sqrd_diff_source / self.sqrd_norm_source),
            )
        )


class Reset(Coupler):
    """The purpose of this coupler is to zero out the saved mass sources."""

    def __init__(self, n_space, opmcase: str):
        self.source = np.zeros(n_space)
        self.opmcase = opmcase
        self.str = "Reset"
        self.num_cells = n_space

    def save_source(self, _, current_step):
        tpysa.write_vtk(
            {"vol_source": self.source}, self.opmcase, current_step, self.num_cells
        )

    def get_source(self, *_):
        return self.source

    def cleanup(self):
        logging.error("Zeroed out the source terms in the vtu files")
