import os
from vtkmodules.vtkIOXML import (
    vtkXMLUnstructuredGridReader,
    vtkXMLUnstructuredGridWriter,
)
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa


def write_vtk(
    solutions: dict,
    opmcase: str,
    current_step: int,
    num_cells: int,
):
    file_name = recover_file_name(opmcase, current_step)

    dataset = extract_dataset(file_name)

    for name, variable in solutions.items():
        if len(variable) == 3 * num_cells:
            variable = variable.reshape((-1, 3), order="F")
        dataset.CellData.append(variable, name)

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(dataset.VTKObject)
    writer.SetDataModeToAscii()
    writer.Write()


def recover_file_name(opmcase: str, current_step: int):
    file_name = os.path.join(
        os.path.dirname(opmcase), "solution", os.path.basename(opmcase)
    )
    file_name += "-{:05}.vtu".format(current_step)

    return file_name


def extract_dataset(file_name):
    if os.path.isfile(file_name):
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()
        vtk_object = reader.GetOutput()

    else:
        raise FileNotFoundError(
            'No VTK files found, first do a run with "OPM" as vtk-writer.'
        )

    dataset = dsa.WrapDataObject(vtk_object)
    return dataset


def read_source_from_vtk(
    opmcase: str,
    current_step: int,
    num_cells: int,
):
    file_name = recover_file_name(opmcase, current_step)
    dataset = extract_dataset(file_name)

    if "vol_source" in dataset.CellData.keys():
        return np.array(dataset.CellData["vol_source"])
    else:
        return np.zeros(num_cells)
