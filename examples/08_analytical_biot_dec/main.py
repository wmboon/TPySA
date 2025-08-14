import os
import sys

import logging
import numpy as np

import tpysa
from true_sol import p_func, source_func, u_func, body_force_func, r_func
from opm.simulators import OnePhaseSimulator


def main(nx=10, write_errors=False):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "nx": nx,
            "n_time": 2,
            "n_total_cells": nx**3,
            "mu": 0.01,
            "lambda": 1,
            "alpha": 0,
            "vtk_writer": "OPM",  # First run with "OPM", then "Python"
            "vtk_reset": False,
        }
    )
    coupler = ConvergenceLagged

    ## Create a n x n x n Cartesian grid
    case_str = "GRID_" + str(nx)
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = CartBiot_Model(
        opmcase,
        data,
        CartGrid,
        CouplerType=coupler,
        SimulatorType=OnePhaseSimulator,
    )
    model.simulate()


class CartBiot_Model(tpysa.Biot_Model):
    """docstring for Biot_CartGrid."""

    def generate_deck(self):
        nx = self.data["nx"]
        self.data.update(
            {
                # "nx": nx,
                "ny": nx,
                "nz": nx,
                "hx": 1 / nx,
                "hy": 1 / nx,
                "hz": 1 / nx,
            }
        )

        dir_name = os.path.dirname(__file__)
        template_file = os.path.join(dir_name, "template/CARTGRID.DATA")
        tpysa.generate_deck_from_template(template_file, self.deck_file, self.data)

    def perform_one_time_step(self, solid_p0: np.ndarray) -> np.ndarray:
        current_step = self.sim.current_step()
        logging.debug("\nReport step {}".format(current_step))

        reportsteps = self.schedule.reportsteps
        dt = (reportsteps[current_step] - reportsteps[current_step - 1]).total_seconds()

        # Extract current fluid pressure
        fluid_p = self.sim.get_primary_variable("pressure")
        source = source_func(self.grid.cell_centers)

        vol_source = source * self.grid.cell_volumes

        self.coupler.process_source(vol_source)

        if current_step < len(reportsteps) - 1:
            var_dict = tpysa.get_fluidstate_variables(self.sim)
            self.coupler.set_mass_source(
                self.grid, self.schedule, current_step, var_dict
            )

        fluid_p = fluid_p - self.data["ref_pressure"]
        fluid_p -= (
            self.grid.cell_volumes * fluid_p
        ).sum() / self.grid.cell_volumes.sum()

        # Solve the mechanics equations
        bf = body_force_func(self.grid.cell_centers)
        displ, rotat, solid_p = self.disc.solve_body_force(
            self.data, bf.ravel(), self.solver
        )

        p_ex = p_func(self.grid.cell_centers)
        p_ex -= (self.grid.cell_volumes * p_ex).sum() / self.grid.cell_volumes.sum()
        displ_ex = u_func(self.grid.cell_centers).ravel()
        rotat_ex = r_func(self.grid.cell_centers).ravel()

        def norm(x):
            weight = np.tile(self.grid.cell_volumes, x.size // self.grid.num_cells)
            return np.sqrt(np.dot(x, weight * x))

        err_u = norm(displ - displ_ex) / norm(displ_ex)
        err_r = norm(rotat - rotat_ex) / norm(rotat_ex)
        err_ps = norm(solid_p)
        err_pf = norm(fluid_p - p_ex) / norm(p_ex)

        logging.warning(
            "nx: {:}, displacement error {:.2e}".format(self.data["nx"], err_u)
        )
        logging.warning("nx: {:}, rotation error {:.2e}".format(self.data["nx"], err_r))
        logging.warning("nx: {:}, p_solid error {:.2e}".format(self.data["nx"], err_ps))
        logging.warning("nx: {:}, p_fluid error {:.2e}".format(self.data["nx"], err_pf))

        self.coupler.save_errs(self.data["nx"], (err_u, err_r, err_ps, err_pf))

        if self.vtk_writer_is_python:
            self.write_vtk(current_step, fluid_p, displ, rotat, solid_p, vol_source)

        return solid_p


class CartGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()

        # Put zero displacements on the boundary
        self.tags["displ_bdry"] = self.tags["domain_boundary_faces"].copy()


class ConvergenceLagged(tpysa.Lagged):
    def save_errs(self, nx, errs):
        self.errs = errs
        self.nx = nx

    def cleanup(self):
        if write_errors == True:
            dir_name = os.path.dirname(__file__)
            with open(os.path.join(dir_name, "errs.txt"), "a") as f:
                str_tuple = [str(self.nx)]
                str_tuple.extend(["{:.4e}".format(err) for err in self.errs])
                f.write(" ".join(str_tuple) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nx = int(sys.argv[1])
        write_errors = True

    else:
        nx = 10
        write_errors = False

    main(nx, write_errors)
