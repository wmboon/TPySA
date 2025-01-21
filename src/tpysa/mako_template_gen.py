import os
import numpy as np
import mako.template


def generate_cart_grid(dimensions: int, template_name="CARTGRID"):

    if isinstance(dimensions, np.ScalarType):
        nx = ny = nz = dimensions
    elif len(dimensions) == 3:
        nx, ny, nz = dimensions
    else:
        raise ValueError

    hx, hy, hz = 1 / np.array([nx, ny, nz])

    data = {"nx": nx, "ny": ny, "nz": nz, "hx": hx, "hy": hy, "hz": hz}

    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "grid_templates/")
    data_file = os.path.join(dir_name, "{:}.DATA".format(template_name))

    template = mako.template.Template(filename=data_file)

    output_file = os.path.join(dir_name, "{:}_{:}.DATA".format(template_name, nx))
    f = open(output_file, "w")
    f.write(template.render(**data))
    f.close()


if __name__ == "__main__":
    generate_cart_grid(5)
