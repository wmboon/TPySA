import os
import numpy as np
import scipy.optimize as spo


class FixedPoint:
    def __init__(
        self,
        main_file: str,
        n_elements: int,
        n_timesteps: int,
        rtol: float = 1e-4,
    ):
        self.n_elements = n_elements
        self.n_timesteps = n_timesteps

        self.python = "/home/AD.NORCERESEARCH.NO/wibo/opm/opm-common/.venv/bin/python"
        self.main = main_file

        self.iteration = 0
        self.residuals = []

        self.rtol = rtol
        self.initial_guess = np.zeros(self.n_elements * (self.n_timesteps + 1))

    def generate_output_dir(self, scheme_name):
        dir_name = os.path.dirname(self.main)
        self.output_dir = os.path.join(dir_name, scheme_name)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def run_simulator(self, input_source: np.ndarray) -> np.ndarray:
        # Save the input source
        source_file = os.path.join(
            self.output_dir, "source_{:}.npz".format(self.iteration)
        )
        np.savez(source_file, psi=input_source.reshape((-1, self.n_elements)))

        # Run the simulation
        os.system(" ".join((self.python, self.main, source_file)))

        # Load the output source
        out_file = list(os.path.splitext(source_file))
        out_file[0] = out_file[0] + "_out"
        out_file = "".join(out_file)
        out_source = np.load(out_file)["psi"]

        self.residuals.append(
            (np.linalg.norm(out_source.ravel() - input_source))
            / np.linalg.norm(out_source)
        )

        print(
            "Fixed point iteration {:}/{:}".format(
                self.iteration + 1, self.n_iterations
            )
        )

        self.iteration += 1

        return out_source.ravel() - input_source

    def anderson(self, n_iterations):
        self.n_iterations = n_iterations
        self.generate_output_dir("anderson")

        psi = spo.anderson(
            self.run_simulator,
            self.initial_guess,
            alpha=1,
            iter=n_iterations,
            maxiter=n_iterations + 2,
            f_rtol=self.rtol,
            verbose=True,
            tol_norm=np.linalg.norm,
        )

        self.save_final_sol(psi)

    def fixed_point(self, n_iterations):
        self.n_iterations = n_iterations
        self.generate_output_dir("fixed_point")

        psi = self.initial_guess.copy()

        for _ in range(n_iterations):
            diff = self.run_simulator(psi)

            psi += diff

            if np.linalg.norm(diff) / np.linalg.norm(psi) < self.rtol:
                break

        self.save_final_sol(psi)

    def save_final_sol(self, psi):
        output_file = os.path.join(self.output_dir, "source_true.npz")
        np.savez(output_file, psi=psi, res=np.array(self.residuals))
