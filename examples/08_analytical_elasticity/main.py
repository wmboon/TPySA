import os

import logging
import numpy as np

import tpysa
from true_sol import u_func, body_force_func, r_func


def main(nx=10):
    ## Input: Model and discretization parameters
    data = tpysa.default_data()
    data.update(
        {
            "inj_rate": 0.1,  # sm3/day
            "nx": nx,
            "n_time": 1,
            "n_total_cells": nx**3,
            "mu": 0.01,
            "lambda": 1,
            "alpha": 0,
            "vtk_writer": "OPM",  # First run with "OPM", then "Python"
            "vtk_reset": False,
        }
    )
    coupler = tpysa.Lagged

    ## Create a n x n x n Cartesian grid
    case_str = "GRID_" + str(nx)
    opmcase = tpysa.opmcase_from_main(__file__, case_str)

    model = CartBiot_Model(
        opmcase,
        data,
        CartGrid,
        CouplerType=coupler,
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
        # Solve the mechanics equations
        bf = body_force_func(self.grid.cell_centers)

        displ, rotat, solid_p = self.disc.solve_body_force(
            self.data, bf.ravel(), self.solver
        )

        displ_ex = u_func(self.grid.cell_centers).ravel()
        rotat_ex = r_func(self.grid.cell_centers).ravel()

        def norm(x):
            weight = np.tile(self.grid.cell_volumes, x.size // self.grid.num_cells)
            return np.sqrt(np.dot(x, weight * x))

        err_u = norm(displ - displ_ex) / norm(displ_ex)
        err_r = norm(rotat - rotat_ex) / norm(rotat_ex)
        err_p = norm(solid_p)

        logging.warning(
            "nx: {:}, displacement error {:.2e}".format(self.data["nx"], err_u)
        )
        logging.warning("nx: {:}, rotation error {:.2e}".format(self.data["nx"], err_r))
        logging.warning("nx: {:}, p_solid error {:.2e}".format(self.data["nx"], err_p))

        fluid_p = np.zeros(self.grid.num_cells)
        self.write_vtk(0, fluid_p, displ, rotat, solid_p, fluid_p)

        return solid_p


class CartGrid(tpysa.Grid):
    """docstring for CartGrid."""

    def tag_boundaries(self):
        super().tag_boundaries()

        # Put zero displacements on the boundary
        self.tags["displ_bdry"] = self.tags["domain_boundary_faces"].copy()


if __name__ == "__main__":
    main(40)
