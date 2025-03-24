from opm.simulators import GasWaterSimulator, BlackOilSimulator
from typing import Union


def get_fluidstate_variables(sim: Union[GasWaterSimulator, BlackOilSimulator]):
    keys = [
        # "po",
        "pg",
        "pw",
        "rho_w",
        "rho_g",
        # "rho_o",
        # "Rs",
        # "Rv",
        "Sw",
        # "So",
        "Sg",
        "T",
    ]

    return {key: sim.get_fluidstate_variable(key) for key in keys}
