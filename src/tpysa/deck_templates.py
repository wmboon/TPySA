import mako.template
import numpy as np


def generate_deck_from_template(template_file: str, output_file: str, data: dict):
    data["rockbiot_str"] = stringify_rockbiot(data["rock_biot"], data["n_total_cells"])
    template = mako.template.Template(filename=template_file)

    with open(output_file, "w") as f:
        f.write(template.render(**data))


def stringify_rockbiot(rockbiot: np.ndarray, num_cells: int):
    # Make rockbiot into a string
    rockbiot = rockbiot * 1e5  # Conversion from 1/Pa to 1/bar

    if isinstance(rockbiot, np.ndarray):
        if np.all(rockbiot == rockbiot[0]):
            rockbiot_str = "{}*{}".format(num_cells, rockbiot[0])
        else:
            rockbiot_str = " ".join([str(rock) for rock in rockbiot])
    elif isinstance(rockbiot, np.ScalarType):
        rockbiot_str = "{}*{}".format(num_cells, rockbiot)
    else:
        raise TypeError

    return rockbiot_str
