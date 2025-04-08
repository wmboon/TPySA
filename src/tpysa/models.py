import numpy as np
import os

from opm.simulators import BlackOilSimulator
from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.util import EModel

import tpysa


class Biot_Model:
    def __init__(
        self,
        opmcase: str,
        data: dict,
        GridType=tpysa.Grid,
        SimulatorType=BlackOilSimulator,
        SolverType=None,
        CouplerType=tpysa.Lagged,
    ):
        self.opmcase = opmcase
        self.deck_file = "{}.DATA".format(self.opmcase)

        self.data = data
        self.GridType = GridType
        self.SimulatorType = SimulatorType
        self.SolverType = SolverType
        self.CouplerType = CouplerType

        vtk_flag = self.data.get("vtk_writer", "Python")
        self.py_write_vtk = vtk_flag == "Python"

    def simulate(self):
        self.initialize()
        self.run()

    def initialize(self):
        # Compute the rock_biot scaling in units of 1/Pa
        self.compute_rock_biot()

        # Generate and parse deck
        self.generate_deck()
        deck = Parser().parse(self.deck_file)

        # Create the folder for the solution if it does not exist
        output_dir = os.path.join(os.path.dirname(self.deck_file), "solution")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            self.py_write_vtk = 0

        ## Initialize flow simulator
        state = EclipseState(deck)
        self.schedule = Schedule(deck, state)
        summary_config = SummaryConfig(deck, state, self.schedule)
        self.sim = self.SimulatorType(
            deck,
            state,
            self.schedule,
            summary_config,
            args=[
                "--enable-vtk-output={}".format(1 - self.py_write_vtk),
                "--output-dir={}".format(output_dir),
            ],
        )

        self.operate_wells(self.schedule)

        ## Extract grid
        self.sim.step_init()  # Creates the EGRID and INIT files

        case_name = os.path.basename(self.opmcase)

        egrid_file = os.path.join(output_dir, "{}.EGRID".format(case_name))
        self.grid = self.GridType(egrid_file)

        ## Extract Emodel
        init_file = os.path.join(output_dir, "{}.INIT".format(case_name))
        self.emodel = EModel(init_file)
        self.data["FIPNUM"] = self.emodel.get("FIPNUM")

        # Double check that ROCKBIOT is inserted appropriately
        field_props = state.field_props()
        rock_biot_ecl = field_props["ROCKBIOT"]
        assert np.allclose(self.data["rock_biot"], rock_biot_ecl)

        # Manage the physical parameters
        self.manage_data(self.grid.num_cells)

        ## Initialize Mechanics
        self.data["ref_pressure"] = self.sim.get_primary_variable("pressure")
        self.disc = tpysa.TPSA(self.grid)
        self.disc.discretize(self.data)

        ## Choose Solver
        if self.SolverType is None:
            if self.disc.ndofs.sum() <= 1e4:
                self.SolverType = tpysa.DirectSolver
            else:
                self.SolverType = tpysa.AMGSolver
        self.solver = self.SolverType(self.disc.system)

        ## Choose coupling scheme
        n_time = len(self.schedule.reportsteps)
        n_space = self.grid.num_cells

        self.coupler = self.CouplerType(n_space, n_time, self.opmcase)

    def run(self):
        ## Initial conditions
        solid_p0 = np.zeros_like(self.data["ref_pressure"])

        reportsteps = self.schedule.reportsteps

        ## Ready to simulate
        while not self.sim.check_simulation_finished():
            current_step = self.sim.current_step()
            dt = (
                reportsteps[current_step] - reportsteps[current_step - 1]
            ).total_seconds()

            var_dict = tpysa.get_fluidstate_variables(self.sim)

            # Compute current fluid and solid pressures
            fluid_p = self.sim.get_primary_variable("pressure")
            displ, rotat, solid_p = self.disc.solve(self.data, fluid_p, self.solver)

            # Compute the change in solid pressures
            # during the previous time step
            delta_ps = (solid_p - solid_p0) / dt

            # Compute the mass source
            vol_source = -self.data["alpha"] / self.data["lambda"] * delta_ps

            # Set the mass source
            self.coupler.save_source(current_step, vol_source)
            self.coupler.set_mass_source(
                self.grid, self.schedule, current_step, var_dict
            )

            # Save the solid pressure for the next time step
            solid_p0 = solid_p.copy()

            # Output solution
            self.write_vtk(current_step, fluid_p, displ, rotat, solid_p)

            # Advance
            self.sim.step()

        ## Compute solution at the final step
        fluid_p = self.sim.get_primary_variable("pressure")
        displ, rotat, solid_p = self.disc.solve(self.data, fluid_p, self.solver)
        self.write_vtk(self.sim.current_step(), fluid_p, displ, rotat, solid_p)

        ## Cleanup
        self.coupler.cleanup()
        self.sim.step_cleanup()

    def write_vtk(
        self,
        current_step: int,
        fluid_p: np.ndarray,
        displ: np.ndarray,
        rotat: np.ndarray,
        solid_p: np.ndarray,
    ):
        if not self.py_write_vtk:
            return
        else:
            vol_change = self.disc.recover_volumetric_change(
                solid_p, fluid_p, self.data
            )
            diff_p = fluid_p - self.data["ref_pressure"]
            sol_dict = {
                "pressure_fluid": fluid_p,
                "pressure_solid": solid_p,
                "displacement": displ,
                "rotation": rotat,
                "vol_change": vol_change,
                "pressure_diff": diff_p,
                "FIPNUM": self.data["FIPNUM"],
            }
            tpysa.write_vtk(self.grid, sol_dict, self.opmcase, current_step)

    def compute_rock_biot(self):
        self.data["rock_biot"] = (
            self.data["alpha"] * self.data["alpha"] / self.data["lambda"]
        )

        assert isinstance(self.data["rock_biot"], np.ScalarType)

    def manage_data(self, num_cells: int):
        """
        Ensure the data entries are cell-wise
        """
        data = self.data

        for key in ["lambda", "mu", "alpha"]:
            if isinstance(data[key], np.ScalarType):
                data[key] = np.full(num_cells, data[key])

    def generate_deck(self):
        dir_name, deck_name = os.path.split(self.deck_file)
        template_file = os.path.join(dir_name, "template", deck_name)

        tpysa.generate_deck_from_template(template_file, self.deck_file, self.data)

    def operate_wells(self, schedule: Schedule):
        pass
