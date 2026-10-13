"""Offscreen VTK brush verification. Run: python -m unittest -v test_brush.py"""
import types
import unittest
from pathlib import Path
import numpy as np
import vtk
from biocal3d_annotator import SurfaceView
from mesh_io import Mesh, polydata

class BrushTests(unittest.TestCase):
    def test_visible_surface_only_and_roi_guard(self):
        # Two identical grids, back surface directly behind front surface.
        n=11; x,y=np.meshgrid(np.linspace(-2,2,n),np.linspace(-2,2,n))
        front=np.column_stack((x.ravel(),y.ravel(),np.zeros(n*n)))
        v=np.vstack((front,front-[0,0,1]))
        f=[]
        for j in range(n-1):
            for i in range(n-1):
                a=j*n+i;f.extend([(a,a+1,a+n),(a+1,a+n+1,a+n)])
        f=np.asarray(f); count=len(f); f=np.vstack((f,f+n*n))
        p=polydata(v,f)
        active=np.ones(len(f),bool); active[:count//2]=False
        mesh=Mesh(Path('test.ply'),v,f,p,active)
        view=types.SimpleNamespace(mesh=mesh,projection_key=None,projected=None,centers=v[f].mean(axis=1),overlay_enabled=True)
        rw=vtk.vtkRenderWindow();rw.SetOffScreenRendering(1);rw.SetSize(500,500);rw.SetMultiSamples(0)
        renderer=vtk.vtkRenderer();rw.AddRenderer(renderer);view.renderer=renderer
        mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(p)
        view.base=vtk.vtkActor();view.base.SetMapper(mapper);renderer.AddActor(view.base)
        view.overlay=vtk.vtkActor();view.overlay.VisibilityOff()
        view.circle=vtk.vtkActor2D();view.circle.VisibilityOff()
        view.GetRenderWindow=lambda:rw
        view._project_centers=types.MethodType(SurfaceView._project_centers,view)
        locator=vtk.vtkStaticCellLocator();locator.SetDataSet(p);locator.BuildLocator()
        view.picker=vtk.vtkCellPicker();view.picker.PickFromListOn();view.picker.AddPickList(view.base);view.picker.AddLocator(locator)
        camera=renderer.GetActiveCamera();camera.SetPosition(0,0,10);camera.SetFocalPoint(0,0,0);camera.SetViewUp(0,1,0)
        camera.ParallelProjectionOn();camera.SetParallelScale(2.5);renderer.ResetCameraClippingRange();rw.Render()
        selected=SurfaceView.brush_faces(view,250,270,70)
        self.assertGreater(len(selected),5)
        self.assertTrue(np.all(selected<count),'Brush must not label hidden back surface')
        self.assertTrue(active[selected].all(),'Brush must not modify outside-ROI faces')
        projected=view._project_centers()
        dist=np.linalg.norm(projected[selected]-[250,270],axis=1)
        self.assertTrue((dist<=70).all(),'Brush must be circular')
        rw.Finalize()

if __name__=='__main__':unittest.main()
