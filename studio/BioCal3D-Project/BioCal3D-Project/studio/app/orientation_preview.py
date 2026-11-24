"""Load full original scanner geometry in its verified canonical ROI frame."""
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
import orient_scanners as core
from mesh_io import load_mesh


def preview_model(root,row,originals):
    root=Path(root)
    g=core.arrays(root/row['sample_dir']/'geometry.npz')
    hint=str(g.get('source_ply',row.get('source','')));checksum=str(g.get('source_sha256',''))
    sample=row.get('sample_id',Path(row['sample_dir']).name)
    path=originals.resolve([hint],scanner=row['scanner'],sample=sample,checksum=checksum)
    mesh=load_mesh(path)
    poly=vtk.vtkPolyData();poly.DeepCopy(mesh.display)
    vertices=core.points(vtk_to_numpy(poly.GetPoints().GetData()),np.asarray(g['original_to_canonical'],float))
    poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(vertices),deep=True))
    poly.GetPointData().SetNormals(None);poly.GetCellData().SetNormals(None)
    association='none'
    for data,kind in ((poly.GetPointData(),'point'),(poly.GetCellData(),'cell')):
        rgb=data.GetScalars()
        if rgb is not None and rgb.GetNumberOfComponents() in (3,4):association=kind;break
    texture=None
    if mesh.texture and poly.GetPointData().GetTCoords() is not None:
        factory=vtk.vtkImageReader2Factory();reader=factory.CreateImageReader2(str(mesh.texture))
        if reader:
            reader.SetFileName(str(mesh.texture));reader.Update()
            texture=vtk.vtkTexture();texture.SetInputConnection(reader.GetOutputPort());texture.InterpolateOn()
    return {'poly':poly,'association':association,'texture':texture,'bounds':list(poly.GetBounds()),
            'path':str(path),'appearance':'texture RGB' if texture else ('vertex RGB' if association!='none' else 'geometry only'),
            'sample_id':sample,'source_sha256':core.sha(path)}
