import numpy as np


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
