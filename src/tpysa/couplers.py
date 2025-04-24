import logging

import numpy as np
from opm.io.ecl import EGrid
from opm.io.schedule import Schedule

import tpysa


class Coupler:
    def set_mass_source(
        self, grid: EGrid, schedule: Schedule, current_step: int, variables: dict
    ) -> None:
        source = self.get_source(current_step).astype(float, copy=True)

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

    def source_to_str(self, grid: EGrid, source: np.ndarray) -> str:
        output = [""] * len(source)
        for c, s in enumerate(source):
            ijk = [i + 1 for i in grid.ijk_from_active_index(c)]
            output[c] = "\t {} {} {} WATER {} /".format(*ijk, s)

        return "\n".join(["SOURCE", *output, "/"])


class Lagged(Coupler):
    """docstring for Lagged coupler."""

    def __init__(self, volumes, *args):
        self.source = np.zeros_like(volumes)
        self.str = "lagged"

    def process_source(self, source, *args) -> None:
        self.source = source

    def get_source(self, *args) -> None:
        return self.source

    def cleanup(self) -> None:
        pass


class Iterative(Coupler):
    """docstring for Iterative coupler."""

    def __init__(self, volumes: np.ndarray, opmcase: str):
        self.source = np.zeros_like(volumes)
        self.opmcase = opmcase
        self.str = "iterative"

        self.volumes = volumes
        self.sqrd_diff_source = 0.0
        self.sqrd_norm_source = 0.0
        self.initialize_logger()

    def initialize_logger(self) -> None:
        logger = logging.getLogger()
        ch = logging.FileHandler(self.opmcase + ".ITER")
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    def process_source(self, source, dt: float) -> None:
        """
        Compares the computed source to the one from the previous space-time iteration
        """
        diff = source - self.source
        self.sqrd_diff_source += dt * np.dot(diff, self.volumes * diff)
        self.sqrd_norm_source += dt * np.dot(source, self.volumes * source)

    def get_source(self, current_step: int) -> np.ndarray:
        """
        Extracts the source for (t_i, t_{i + 1}] from the vtu-file at t_{i + 1}
        """
        self.source = tpysa.read_source_from_vtk(
            self.opmcase, current_step + 1, self.volumes.size
        )
        return self.source

    def cleanup(self) -> None:
        logging.error(
            "Source difference, abs: {:.2e}, rel: {:.4e}".format(
                np.sqrt(self.sqrd_diff_source),
                np.sqrt(self.sqrd_diff_source / self.sqrd_norm_source),
            )
        )
