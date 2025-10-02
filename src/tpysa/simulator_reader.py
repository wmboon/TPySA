from typing import Union

from opm.simulators import BlackOilSimulator, GasWaterSimulator, OnePhaseSimulator


def get_fluidstate_variables(
    sim: Union[
        OnePhaseSimulator,
        GasWaterSimulator,
        BlackOilSimulator,
    ],
) -> dict:
    keys = [
        # "po",
        # "pg",
        # "pw",
        "rho_w",
        # "rho_g",
        # "rho_o",
        # "Rs",
        # "Rv",
        "Sw",
        # "So",
        # "Sg",
        "T",
    ]

    return {key: sim.get_fluidstate_variable(key) for key in keys}
