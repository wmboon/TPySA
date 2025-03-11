import os
import numpy as np
import mako.template


def generate_cart_grid(
    num_cells: int,
    domain_length: float = 100,
    template_name: str = "CARTGRID",
    output_file: str = None,
    rockbiot: np.ndarray = None,
    inj_rate: float = 0.0,
    time_steps: int = 20,
):

    if isinstance(num_cells, np.ScalarType):
        nx = ny = nz = num_cells
    elif len(num_cells) == 3:
        nx, ny, nz = num_cells
    else:
        raise ValueError

    # Make rockbiot into a string
    if rockbiot is None:
        rockbiot_str = "{}*{}".format(nx * ny * nz, 0.0)
    else:
        rockbiot = rockbiot * 1e5  # Conversion from 1/Pa to 1/bar
        rockbiot_str = "{}*{}".format(nx * ny * nz, rockbiot[0])
        # rockbiot_str = str(rockbiot)[1:-1]

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

    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "grid_templates/")
    template_file = os.path.join(dir_name, "{:}.DATA".format(template_name))

    template = mako.template.Template(filename=template_file)

    if output_file is None:
        output_file = os.path.join(dir_name, "{:}_{:}.DATA".format(template_name, nx))

    f = open(output_file, "w")
    f.write(template.render(**data))
    f.close()


if __name__ == "__main__":
    generate_cart_grid(5)
