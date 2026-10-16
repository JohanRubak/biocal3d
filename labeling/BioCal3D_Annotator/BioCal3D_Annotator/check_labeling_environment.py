"""Check the installed helper modules and real Qt/VTK GUI initialization."""
from pathlib import Path
import sys

annotator_dir = Path(__file__).resolve().parent / "biocal3d_workflow" / "annotator"
sys.path.insert(0, str(annotator_dir))
import numpy
import PySide6
from PySide6 import QtCore, QtWidgets
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import mesh_io
import annotation_features

app = QtWidgets.QApplication([])
widget = QVTKRenderWindowInteractor()
widget.resize(320, 240)
widget.show()
widget.Initialize()
app.processEvents()
widget.GetRenderWindow().Render()
app.processEvents()
widget.Finalize()
widget.close()
print("PASS: helper imports, Windows GUI startup, and VTK rendering")
print("Python:", sys.executable)
print("Qt:", QtCore.qVersion(), "VTK:", vtk.vtkVersion.GetVTKVersion(), "NumPy:", numpy.__version__)
