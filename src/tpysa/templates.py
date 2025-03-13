import os
import numpy as np
import mako.template


def generate_cart_grid(
    num_cells: int,
    domain_length: float = 100,
    output_file: str = None,
    rockbiot: np.ndarray = None,
    inj_rate: float = 0.0,
    time_steps: int = 20,
):
    template_name = "CARTGRID"

    if isinstance(num_cells, np.ScalarType):
        nx = ny = nz = num_cells
    elif len(num_cells) == 3:
        nx, ny, nz = num_cells
    else:
        raise ValueError

    rockbiot_str = stringify_rockbiot(rockbiot, nx * ny * nz)

    data = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "hx": domain_length / nx,
        "hy": domain_length / ny,
        "hz": domain_length / nz,
        "n_time": time_steps,
        "rockbiot": rockbiot_str,
        "inj_rate": inj_rate,  # m3/day
    }

    generate_grid_from_template(template_name, output_file, data)


def generate_faulted_grid(
    rockbiot: np.ndarray = None,
    output_file: str = None,
    time_steps: int = 20,
    inj_rate: float = 0.0,
):
    template_name = "FAULTGRID"

    rockbiot_str = stringify_rockbiot(rockbiot, 2700)

    data = {
        "n_time": time_steps,
        "rockbiot": rockbiot_str,
        "inj_rate": inj_rate,  # m3/day
    }

    generate_grid_from_template(template_name, output_file, data)


def stringify_rockbiot(rockbiot: np.ndarray, num_cells: int):
    # Make rockbiot into a string
    if isinstance(rockbiot, np.ndarray):
        rockbiot = rockbiot * 1e5  # Conversion from 1/Pa to 1/bar

        if np.all(rockbiot == rockbiot[0]):
            rockbiot_str = "{}*{}".format(num_cells, rockbiot[0])
        else:
            rockbiot_str = " ".join([str(rock) for rock in rockbiot])
    else:
        rockbiot_str = "{}*{}".format(num_cells, 0.0)

    return rockbiot_str


def generate_grid_from_template(template_name: str, output_file: str, data: dict):
    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "grid_templates/")
    template_file = os.path.join(dir_name, "{:}.DATA".format(template_name))

    template = mako.template.Template(filename=template_file)

    if output_file is None:
        output_file = os.path.join(dir_name, "{:}_gen.DATA".format(template_name))

    with open(output_file, "w") as f:
        f.write(template.render(**data))


if __name__ == "__main__":
    generate_cart_grid(5)
