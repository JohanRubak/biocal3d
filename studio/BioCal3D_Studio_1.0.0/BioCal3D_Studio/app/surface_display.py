"""Explicit display subsets with exact source-face picking indices."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray,vtk_to_numpy


def display_subset(data,face_ids):
    face_ids=np.asarray(face_ids,np.int64).reshape(-1)
    if np.any(face_ids<0) or np.any(face_ids>=data.GetNumberOfCells()) or len(np.unique(face_ids))!=len(face_ids):
        raise ValueError('Invalid display face selection.')
    if len(face_ids)==data.GetNumberOfCells() and np.array_equal(face_ids,np.arange(len(face_ids))):
        return data,face_ids.copy()
    if not len(face_ids):return vtk.vtkPolyData(),face_ids.copy()
    copy=vtk.vtkPolyData();copy.ShallowCopy(data)
    a=numpy_to_vtkIdTypeArray(np.arange(data.GetNumberOfCells(),dtype=np.int64),deep=True);a.SetName('ReviewDisplayFaceId');copy.GetCellData().AddArray(a)
    node=vtk.vtkSelectionNode();node.SetFieldType(vtk.vtkSelectionNode.CELL);node.SetContentType(vtk.vtkSelectionNode.INDICES)
    node.SetSelectionList(numpy_to_vtkIdTypeArray(face_ids,deep=True))
    selection=vtk.vtkSelection();selection.AddNode(node)
    extract=vtk.vtkExtractSelection();extract.SetInputData(0,copy);extract.SetInputData(1,selection);extract.Update()
    surface=vtk.vtkDataSetSurfaceFilter();surface.SetInputConnection(extract.GetOutputPort());surface.Update()
    result=vtk.vtkPolyData();result.DeepCopy(surface.GetOutput())
    mapped=vtk_to_numpy(result.GetCellData().GetArray('ReviewDisplayFaceId')).copy().astype(np.int64)
    if len(mapped)!=len(face_ids) or set(mapped)!=set(face_ids):raise ValueError('Display extraction changed face correspondence.')
    return result,mapped
