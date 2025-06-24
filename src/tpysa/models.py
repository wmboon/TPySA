import logging
import os

import numpy as np
from opm.io.ecl_state import EclipseState
from opm.io.parser import Parser
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.simulators import BlackOilSimulator
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
        self.tpsa_output_file = "{}_TPSA.INFO".format(self.opmcase)

        self.data = data
        self.GridType = GridType
        self.SimulatorType = SimulatorType
        self.SolverType = SolverType
        self.CouplerType = CouplerType

        vtk_writer = self.data.get("vtk_writer", "Python")
        self.vtk_writer_is_python = vtk_writer == "Python"
        self.vtk_reset = self.data.get("vtk_reset", False)
        self.initialize_logger()

    def initialize_logger(self) -> None:
        # Logging the debug info
        logging.basicConfig(
            format="%(message)s",
            filename=self.tpsa_output_file,
            filemode="w",
            level=logging.DEBUG,
        )

        # Logging the general information to stdout
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("TPSA: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def simulate(self) -> None:
        self.initialize()
        self.run()

    def initialize(self) -> None:
        # Compute the rock_biot scaling in units of 1/Pa
        self.data["rock_biot"] = self.compute_rock_biot()

        # Generate and parse deck
        self.generate_deck()
        deck = Parser().parse(self.deck_file)

        # Create the folder for the solution if it does not exist
        output_dir = os.path.join(os.path.dirname(self.deck_file), "solution")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            self.vtk_writer_is_python = 0

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
                "--enable-vtk-output={}".format(1 - self.vtk_writer_is_python),
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
        self.data["bdry_mu_over_delta"] = self.compute_spring_constant()

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
        rtol = self.data.get("rtol", 1e-5)
        self.solver = self.SolverType(self.disc.system, rtol)

        ## Choose coupling scheme
        self.coupler = self.CouplerType(self.grid.cell_volumes, self.opmcase)

        if self.vtk_reset:
            self.zero_out_vol_source()

    def zero_out_vol_source(self) -> None:
        num_cells = self.grid.num_cells
        num_steps = len(self.schedule.reportsteps)

        for ind in range(num_steps):
            tpysa.write_vtk(
                {"vol_source": np.zeros(num_cells)}, self.opmcase, ind, num_cells
            )
        logging.error("Reset ON: Zeroed out the source terms in the vtu files")

    def run(self) -> None:
        logging.debug("\nStart of Simulation")

        ## Initial conditions
        solid_p = np.zeros_like(self.data["ref_pressure"])

        ## Ready to simulate
        while not self.sim.check_simulation_finished():
            # Update the solid pressure and mass source
            solid_p = self.perform_one_time_step(solid_p)

            # Advance the flow simulation
            self.sim.step()

        ## Compute solution at the final step
        solid_p = self.perform_one_time_step(solid_p)

        ## Cleanup
        self.coupler.cleanup()
        self.sim.step_cleanup()

    def perform_one_time_step(self, solid_p0: np.ndarray) -> np.ndarray:
        if not self.vtk_writer_is_python:
            logging.debug("Python coupling deactivated.")
            return solid_p0

        else:
            current_step = self.sim.current_step()
            logging.debug("\nReport step {}".format(current_step))

            reportsteps = self.schedule.reportsteps
            dt = (
                reportsteps[current_step] - reportsteps[current_step - 1]
            ).total_seconds()

            # Extract current fluid pressure
            fluid_p = self.sim.get_primary_variable("pressure")

            # Solve the mechanics equations
            displ, rotat, solid_p = self.disc.solve(self.data, fluid_p, self.solver)

            # Compute the change in solid pressures during the previous time step
            delta_ps = (solid_p - solid_p0) / dt

            # Compute the source in the mass balance equation
            vol_source = (
                -self.data["alpha"]
                / self.data["lambda"]
                * delta_ps
                * self.grid.cell_volumes
            )

            # Let the coupler process and set the mass source
            self.coupler.process_source(vol_source, dt)

            if current_step < len(reportsteps) - 1:
                var_dict = tpysa.get_fluidstate_variables(self.sim)
                self.coupler.set_mass_source(
                    self.grid, self.schedule, current_step, var_dict
                )

            # Output solution at time t_i and save the mass source for (t_{i - 1}, t_i]
            self.write_vtk(current_step, fluid_p, displ, rotat, solid_p, vol_source)

            return solid_p

    def write_vtk(
        self,
        current_step: int,
        fluid_p: np.ndarray,
        displ: np.ndarray,
        rotat: np.ndarray,
        solid_p: np.ndarray,
        vol_source: np.ndarray,
    ) -> None:
        vol_change = self.disc.recover_volumetric_change(solid_p, fluid_p, self.data)
        diff_p = fluid_p - self.data["ref_pressure"]
        sol_dict = {
            "pressure_fluid": fluid_p,
            "pressure_solid": solid_p,
            "displacement": displ,
            "rotation": rotat,
            "vol_change": vol_change,
            "pressure_diff": diff_p,
            "FIPNUM": self.data["FIPNUM"],
            "vol_source": vol_source,
        }
        tpysa.write_vtk(sol_dict, self.opmcase, current_step, self.grid.num_cells)

    def compute_rock_biot(self) -> float:
        rock_biot = self.data["alpha"] * self.data["alpha"] / self.data["lambda"]
        assert isinstance(rock_biot, np.ScalarType)

        return rock_biot

    def compute_spring_constant(self) -> np.ndarray:
        mu_delta_scalar = self.data.get("bdry_mu_over_delta")

        if mu_delta_scalar is None:
            delta_typ = 0.05 * (self.grid.nodes[2].max() - self.grid.nodes[2].min())
            mu_typ = np.mean(self.data["mu"])
            mu_delta_scalar = mu_typ / delta_typ

        mu_delta_vec = np.zeros(self.grid.num_faces)
        mu_delta_vec[self.grid.tags["sprng_bdry"]] = mu_delta_scalar
        mu_delta_vec[self.grid.tags.get("tract_bdry", [])] = 0

        return mu_delta_vec

    def manage_data(self, num_cells: int) -> None:
        """
        Ensure the data entries are cell-wise
        """
        data = self.data

        for key in ["lambda", "mu", "alpha"]:
            if isinstance(data[key], np.ScalarType):
                data[key] = np.full(num_cells, data[key])

    def generate_deck(self) -> None:
        dir_name, deck_name = os.path.split(self.deck_file)
        template_file = os.path.join(dir_name, "template", deck_name)

        tpysa.generate_deck_from_template(template_file, self.deck_file, self.data)

    def operate_wells(self, schedule: Schedule) -> None:
        pass
