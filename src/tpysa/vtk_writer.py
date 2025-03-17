import os
import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
import tpysa


def write_vtk(
    grid: tpysa.Grid,
    solutions: dict,
    opmcase: str,
    current_step: int,
):
    file_name = os.path.join(
        os.path.dirname(opmcase), "solution", os.path.basename(opmcase)
    )
    file_name += "-{:05}.vtu".format(current_step)

    if os.path.isfile(file_name):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()
        vtk_object = reader.GetOutput()

        if vtk_object.points is None:  # If the grid is empty
            vtk_object = grid.get_vtk()

    else:
        vtk_object = grid.get_vtk()

    dataset = dsa.WrapDataObject(vtk_object)

    for name, variable in solutions.items():
        if len(variable) == 3 * grid.num_cells:
            variable = variable.reshape((-1, 3), order="F")
        dataset.CellData.append(variable, name)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(dataset.VTKObject)
    writer.SetDataModeToAscii()
    writer.Write()
