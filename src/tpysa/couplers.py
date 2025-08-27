import logging

import numpy as np
from opm.io.ecl import EGrid
from opm.io.schedule import Schedule

import tpysa


class Coupler:
    def __init__(self, volumes: np.ndarray, opmcase: str):
        self.source = np.zeros_like(volumes)
        self.opmcase = opmcase

        self.volumes = volumes

        self.sqrd_true_diff = 0.0
        self.sqrd_true_norm = 0.0

        self.initialize_logger()

    def set_mass_source(
        self, grid: EGrid, schedule: Schedule, current_step: int, rho_w: np.ndarray
    ) -> None:
        source = self.get_source(current_step).astype(float, copy=True)

        # Make into array if it is a scalar
        if isinstance(source, np.ScalarType):
            source = np.full(grid.num_cells, source)

        # Convert units from m3/s to kg/day
        source *= 24 * 60 * 60  # from 1/second to 1/day
        source *= rho_w  # from m^3 to kg

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

    def initialize_logger(self) -> None:
        logger = logging.getLogger()
        ch = logging.FileHandler(self.opmcase + ".ITER")
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    def compare_to_truth(self, source, dt, current_step):
        true_source = tpysa.read_source_from_vtk(
            self.opmcase, current_step, self.volumes.size, True
        )

        diff = source - true_source
        self.sqrd_true_diff += dt * np.sum(diff * self.volumes * diff)
        self.sqrd_true_norm += dt * np.sum(true_source * self.volumes * true_source)

    def print_truth_comparison(self):
        logging.error(
            "Truth comparison {:}, abs: {:.2e}, rel: {:.2e}".format(
                self.str,
                np.sqrt(self.sqrd_true_diff),
                np.sqrt(self.sqrd_true_diff / self.sqrd_true_norm),
            )
        )

    def cleanup(self) -> None:
        pass


class Lagged(Coupler):
    """docstring for Lagged coupler."""

    def __init__(self, volumes, opmcase):
        super().__init__(volumes, opmcase)
        self.str = "lagged"

    def process_source(self, source, **kwargs) -> None:
        self.source = source

    def get_source(self, *args) -> None:
        return self.source


class Iterative(Coupler):
    """docstring for Iterative coupler."""

    def __init__(self, volumes, opmcase):
        super().__init__(volumes, opmcase)
        self.str = "iterative"

        self.sqrd_diff_source = 0.0
        self.sqrd_norm_source = 0.0

    def process_source(self, source, dt: float = 0, **kwargs) -> None:
        """
        Compares the computed source to the one from the previous space-time iteration
        """
        diff = source - self.source
        self.sqrd_diff_source += dt * np.sum(diff * self.volumes * diff)
        self.sqrd_norm_source += dt * np.sum(source * self.volumes * source)

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
            "Source difference, abs: {:.2e}, rel: {:.2e}".format(
                np.sqrt(self.sqrd_diff_source),
                np.sqrt(self.sqrd_diff_source / self.sqrd_norm_source),
            )
        )
