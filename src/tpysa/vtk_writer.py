import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
import tpysa


def write_vtk(grid: tpysa.Grid, variables: list, variable_names: list, file_name: str):
    vtk_object = grid.get_vtk()
    dataset = dsa.WrapDataObject(vtk_object)

    for variable, name in zip(variables, variable_names):
        if len(variable) == 3 * grid.num_cells:
            variable = variable.reshape((-1, 3), order="F")
        dataset.CellData.append(variable, name)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(dataset.VTKObject)
    writer.SetDataModeToAscii()
    writer.Write()
