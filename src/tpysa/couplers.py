import logging
import os
import numpy as np
from opm.io.ecl import EGrid
from opm.io.schedule import Schedule


class Coupler:
    def __init__(self, volumes: np.ndarray, opmcase: str):
        self.source = np.zeros_like(volumes)
        self.opmcase = opmcase

        self.volumes = volumes
        self.pressures = []

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

    def save_pressure(self, pressure):
        self.pressures.append(pressure)

    def cleanup(self) -> None:
        self.pressures = np.vstack(self.pressures)


class Lagged(Coupler):
    """docstring for Lagged coupler."""

    def __init__(self, volumes, opmcase, **kwargs):
        super().__init__(volumes, opmcase)
        self.str = "lagged"
        self.source_list = []

    def process_source(self, source, **kwargs) -> None:
        self.source = source
        self.source_list.append(source)

    def get_source(self, *args) -> None:
        return self.source

    def cleanup(self):
        super().cleanup()
        dirname = os.path.dirname(__file__)
        out_file = os.path.join(dirname, "lagged_source")
        sources = np.vstack(self.source_list)
        np.savez(out_file, psi=sources, pres=self.pressures)


class Iterative(Coupler):
    def __init__(self, volumes, opmcase, mass_source_file=None):
        super().__init__(volumes, opmcase)
        self.source_file = mass_source_file

        self.insource = np.load(self.source_file)["psi"]
        self.outsource = np.zeros_like(self.insource)

    def process_source(self, source, current_step=0, **kwargs) -> None:
        self.outsource[current_step] = source

    def get_source(self, current_step: int) -> np.ndarray:
        """
        Extracts the source for (t_i, t_{i + 1}] from the vtu-file at t_{i + 1}
        """
        return self.insource[current_step + 1]

    def cleanup(self) -> None:
        super().cleanup()
        out_file = list(os.path.splitext(self.source_file))
        out_file[0] = out_file[0] + "_out"
        out_file = "".join(out_file)

        np.savez(out_file, psi=self.outsource, pres=self.pressures)
