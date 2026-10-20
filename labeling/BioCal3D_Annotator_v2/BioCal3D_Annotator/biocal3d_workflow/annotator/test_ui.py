"""Desktop smoke test; needs a working display/OpenGL. Run: python test_ui.py"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import vtk
from PySide6 import QtCore,QtGui,QtWidgets,QtTest
from biocal3d_annotator import Annotator
from mesh_io import create_demo,load_mesh,atomic_npz,AnnotationStore,export_ply
from annotation_features import read_orientation
from roi_review_store import read_reviewed
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy

def run():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root=create_demo(Path(tmp)/'demo')
        for n,info in enumerate([
            dict(sample_id='BCG10T2-2',scanner='TRIOS5',group=10,physical_id='G10_S2',stage='clean'),
            dict(sample_id='BCG2T0-10',scanner='LABscanner',group=2,physical_id='G2_S10',stage='baseline'),
            dict(sample_id='BCG3T1-1',scanner='TRIOS3',group=3,physical_id='G3_S1',stage='partial')],1):
            (root/f'SYNTHETIC_DEMO_{n}'/'metadata.json').write_text(json.dumps(info))
        args=argparse.Namespace(root=None,output=None,annotator='qa',demo=False)
        window=Annotator(args)
        errors=[]
        window.show_error=lambda text:errors.append(text)
        window.show();window.activateWindow();app.setActiveWindow(window);QtTest.QTest.qWait(150)
        from mesh_io import discover
        from roi_review_store import load_review_context,save_review
        geometry_store=AnnotationStore(root,root/'annotations','qa')
        for path in discover(root):
            mesh=load_mesh(path);base,mask,settings,_,raw=load_review_context(geometry_store,mesh)
            save_review(geometry_store,path,base,mask,settings,'approved','UI test fixture','',raw)
        window.load_folder(root);QtTest.QTest.qWait(150)
        assert not errors,errors
        assert window.index==0 and len(window.queue)==3
        original_queue=list(window.queue);selected=window.mesh;labels=window.labels.copy()
        for mode,indices in [('sample',[1,2,0]),('sample_desc',[0,2,1]),('scanner',[1,2,0]),('group',[1,2,0]),('stage',[2,1,0])]:
            window.sort_picker.setCurrentIndex(window.sort_picker.findData(mode))
            assert window.queue==[original_queue[i] for i in indices]
            assert window.mesh is selected and window.queue[window.index]==selected.path
            assert window.list.currentRow()==window.index
            np.testing.assert_array_equal(window.labels,labels)
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('original'))
        assert window.queue==original_queue and window.index==0
        # Slider focus must not steal the held rotation key or paint on the drag.
        center=QtCore.QPoint(window.view.width()//2,window.view.height()//2)
        for slider in list(window.color_controls.values())+[window.radius,window.opacity]:
            value=slider.value();slider.setFocus();app.processEvents()
            QtTest.QTest.mouseClick(slider,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(slider.width()//2,slider.height()//2))
            assert app.focusWidget() is slider
            labels_before=window.labels.copy();camera=window.view.renderer.GetActiveCamera()
            position=np.array(camera.GetPosition())
            QtTest.QTest.keyPress(slider,QtCore.Qt.Key.Key_Space)
            assert window.view.space_down and app.focusWidget() is window.view
            QtTest.QTest.mousePress(window.view,QtCore.Qt.MouseButton.LeftButton,pos=center)
            end=center+QtCore.QPoint(45,25)
            move=QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove,QtCore.QPointF(end),QtCore.QPointF(window.view.mapToGlobal(end)),
                                  QtCore.Qt.MouseButton.NoButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)
            app.sendEvent(window.view,move)
            QtTest.QTest.mouseRelease(window.view,QtCore.Qt.MouseButton.LeftButton,pos=end)
            QtTest.QTest.keyRelease(window.view,QtCore.Qt.Key.Key_Space)
            assert not window.view.space_down and not np.allclose(camera.GetPosition(),position)
            np.testing.assert_array_equal(window.labels,labels_before)
            slider.setValue(value);window.view.reset_camera()
        window.reset_colors()
        # Numeric shortcuts also work outside the viewport; notes retain numeric text.
        window.class_picker.setFocus();app.processEvents()
        QtTest.QTest.keyClick(window.class_picker,QtCore.Qt.Key.Key_2)
        assert window.class_picker.currentData()==0
        QtTest.QTest.keyClick(window.class_picker,QtCore.Qt.Key.Key_3)
        assert window.class_picker.currentData()==-1
        window.notes.setFocus();QtTest.QTest.mouseMove(window.notes,QtCore.QPoint(8,8));app.processEvents();QtTest.QTest.keyClicks(window.notes,'1234')
        assert window.notes.toPlainText()=='1234' and window.class_picker.currentData()==-1
        QtTest.QTest.keyClicks(window.notes,' test');assert window.notes.toPlainText()=='1234 test'
        assert not window.view.space_down
        QtTest.QTest.keyClicks(window.notes,' asdwp')
        assert window.notes.toPlainText().endswith(' asdwp') and window.index==0
        assert window.surface_toggle.isChecked() and not window.points_toggle.isChecked() and not window.wire_toggle.isChecked()
        window.view.setFocus();app.processEvents()
        QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_1,QtCore.Qt.KeyboardModifier.KeypadModifier)
        assert window.class_picker.currentData()==1
        initial_fingerprint=window.mesh.fingerprint
        for shading in range(3):
            window.shading_picker.setCurrentIndex(shading)
            assert bool(window.view.base.GetProperty().GetLighting())==(shading!=0)
        window.shading_picker.setCurrentIndex(0)
        window.surface_toggle.setChecked(False);window.points_toggle.setChecked(True)
        assert window.view.points.GetVisibility() and not window.view.base.GetVisibility()
        assert window.view.points.GetProperty().GetColor() == (1., 1., 0.)
        assert not window.view.points_mapper.GetScalarVisibility()
        assert window.view.points.GetTexture() is None
        window.wire_toggle.setChecked(True);assert window.view.wire.GetVisibility()
        assert window.mesh.fingerprint==initial_fingerprint
        center=QtCore.QPoint(window.view.width()//2,window.view.height()//2)
        QtTest.QTest.mousePress(window.view,QtCore.Qt.MouseButton.LeftButton,pos=center)
        QtTest.QTest.mouseRelease(window.view,QtCore.Qt.MouseButton.LeftButton,pos=center)
        QtTest.QTest.qWait(20)
        assert np.any(window.labels==1),'Mouse brushing did not paint plaque'
        assert window.view.points.GetVisibility() and window.view.wire.GetVisibility() and not window.view.base.GetVisibility()
        window.surface_toggle.setChecked(True);window.points_toggle.setChecked(False);window.wire_toggle.setChecked(False)
        painted=window.labels.copy()
        undo_count=len(window.undo_stack)
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('sample_desc'))
        assert len(window.undo_stack)==undo_count and window.queue[window.index]==window.mesh.path
        np.testing.assert_array_equal(window.labels,painted)
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('original'))
        for focus in (window.color_controls['brightness'],window.view):
            focus.setFocus();app.processEvents()
            QtTest.QTest.keyClick(focus,QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier)
            assert not np.any(window.labels==1),'Ctrl+Z did not undo the annotation'
            QtTest.QTest.keyClick(focus,QtCore.Qt.Key.Key_Y,QtCore.Qt.KeyboardModifier.ControlModifier)
            np.testing.assert_array_equal(window.labels,painted)
            QtTest.QTest.keyClick(focus,QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier)
            QtTest.QTest.keyClick(focus,QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier|QtCore.Qt.KeyboardModifier.ShiftModifier)
            np.testing.assert_array_equal(window.labels,painted)
        window.autosave();assert window.store.status(window.queue[0])=='draft'
        window.notes.setPlainText('Saved before keyboard navigation')
        window.color_controls['brightness'].setFocus();app.processEvents()
        QtTest.QTest.keyClick(window.color_controls['brightness'],QtCore.Qt.Key.Key_D)
        assert window.index==1 and window.store.status(original_queue[0])=='draft'
        assert window.store.load(read_reviewed(window.store,original_queue[0]))[2]=='Saved before keyboard navigation'
        QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_A);assert window.index==0
        np.testing.assert_array_equal(window.labels,painted)
        QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_A);assert window.index==0
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('scanner'))
        assert window.index==2
        QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_D);assert window.index==2
        QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_A);assert window.mesh.path==original_queue[2]
        QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_D);assert window.mesh.path==original_queue[0]
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('original'))
        np.testing.assert_array_equal(window.labels,painted)
        window.fill_clean();assert not np.any(window.labels[window.mesh.active]==-2)
        window.complete_next();assert window.index==1
        assert window.store.status(window.queue[0])=='done'
        assert (window.store.path_for(window.queue[0]).parent/(window.queue[0].name+'.labeled.ply')).exists()
        window.list.setCurrentRow(0);assert window.completed
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('status'))
        assert [window.store.status(p) for p in window.queue]==['draft','draft','done']
        assert window.mesh.path==original_queue[0] and window.index==2 and window.completed
        window.sort_picker.setCurrentIndex(window.sort_picker.findData('original'))
        # Reopened completed annotation becomes a draft after a label edit.
        window.edit(np.array([0]),-1);assert not window.completed
        assert window.store.status(window.queue[0])=='draft'
        # Display adjustments never alter source colors or exported scanner RGB.
        original_bytes=window.mesh.path.read_bytes()
        original_rgb=vtk_to_numpy(window.mesh.display.GetPointData().GetScalars()).copy()
        window.color_controls['brightness'].setValue(20)
        window.color_controls['contrast'].setValue(140)
        window.update_display_colors()
        assert not np.array_equal(vtk_to_numpy(window.view.render_data.GetPointData().GetScalars()),original_rgb)
        np.testing.assert_array_equal(vtk_to_numpy(window.mesh.display.GetPointData().GetScalars()),original_rgb)
        assert window.mesh.path.read_bytes()==original_bytes
        adjusted=vtk_to_numpy(window.view.render_data.GetPointData().GetScalars()).copy();labels=window.labels.copy()
        window.color_toggle.setChecked(False)
        assert not window.view.mapper.GetScalarVisibility() and not window.view.points_mapper.GetScalarVisibility()
        assert window.view.base.GetTexture() is None and window.view.overlay.GetVisibility()
        np.testing.assert_array_equal(window.labels,labels)
        window.save()
        ply=window.store.path_for(window.mesh.path).parent/(window.mesh.path.name+'.labeled.ply')
        np.testing.assert_array_equal(vtk_to_numpy(load_mesh(ply).display.GetPointData().GetScalars()),original_rgb)
        window.color_toggle.setChecked(True);assert window.view.mapper.GetScalarVisibility()
        np.testing.assert_array_equal(vtk_to_numpy(window.view.render_data.GetPointData().GetScalars()),adjusted)
        window.reset_colors()
        np.testing.assert_array_equal(vtk_to_numpy(window.view.render_data.GetPointData().GetScalars()),original_rgb)
        # Bottom view visibly locks the brush, including a zero-thickness surface.
        labels=window.labels.copy();c=window.view.renderer.GetActiveCamera()
        center3d=window.view.surface_center;distance=np.linalg.norm(np.asarray(c.GetPosition())-center3d)
        c.SetPosition(*(center3d-window.view.top_normal*distance))
        window.view.renderer.ResetCameraClippingRange();window.view.GetRenderWindow().Render()
        assert 'BOTTOM' in window.view.side_text.GetInput()
        assert len(window.view.brush_faces(window.view.GetRenderWindow().GetSize()[0]/2,window.view.GetRenderWindow().GetSize()[1]/2,24))==0
        np.testing.assert_array_equal(window.labels,labels)
        window.view.reset_camera();assert 'TOP' in window.view.side_text.GetInput()
        # New rater starts independently; switching back restores the first rater.
        original_rater=window.store.annotator;first_labels=window.labels.copy()
        window.create_annotator('dentist_2');assert window.store.annotator=='dentist_2'
        assert np.all(window.labels[window.mesh.active]==-2)
        window.edit(np.array([0]),1)
        window.rater.setCurrentText(original_rater);np.testing.assert_array_equal(window.labels,first_labels)
        # A screenshot of the real UI with synthetic data, for visual inspection.
        window.view.GetRenderWindow().Render();QtTest.QTest.qWait(100)
        window.screen().grabWindow(window.winId()).save(str(Path(__file__).parent/'preview.png'))
        capture=vtk.vtkWindowToImageFilter();capture.SetInput(window.view.GetRenderWindow());capture.ReadFrontBufferOff();capture.Update()
        writer=vtk.vtkPNGWriter();writer.SetInputConnection(capture.GetOutputPort());writer.SetFileName(str(Path(__file__).parent/'viewport_qa.png'));writer.Write()
        assert not errors,errors
        # UV texture adjustments use a render copy, including automatic PLY exports.
        textured=Path(tmp)/'textured';textured.mkdir()
        image=vtk.vtkImageData();image.SetDimensions(4,4,1)
        pixels=np.tile(np.array([[60,120,180]],np.uint8),(16,1))
        image.GetPointData().SetScalars(numpy_to_vtk(pixels,deep=True))
        tex=textured/'scanner.png';writer=vtk.vtkPNGWriter();writer.SetInputData(image);writer.SetFileName(str(tex));writer.Write()
        atomic_npz(textured/'geometry.npz',vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]]),faces=np.array([[0,1,2],[1,3,2]]),
                   uv=np.array([[0,0],[1,0],[0,1],[1,1]],np.float32),texture_path=np.array(str(tex)))
        window.load_folder(textured);assert window.view.original_texture is not None
        original_texture=tex.read_bytes();original_pixels=window.view.texture_pixels.copy()
        window.color_controls['gamma'].setValue(160);window.color_controls['saturation'].setValue(50);window.update_display_colors()
        assert not np.array_equal(vtk_to_numpy(window.view.adjusted_texture.GetPointData().GetScalars()),original_pixels)
        np.testing.assert_array_equal(vtk_to_numpy(window.view.original_texture.GetPointData().GetScalars()),original_pixels)
        window.color_toggle.setChecked(False)
        assert window.view.base.GetTexture() is window.view.neutral_texture and window.view.points.GetTexture() is None
        assert window.save()
        output=window.store.path_for(window.mesh.path).parent/(window.mesh.path.name+'.labeled.ply')
        assert load_mesh(output).texture.read_bytes()==original_texture and tex.read_bytes()==original_texture
        window.color_toggle.setChecked(True);assert window.view.base.GetTexture() is not None
        window.reset_colors();np.testing.assert_array_equal(vtk_to_numpy(window.view.adjusted_texture.GetPointData().GetScalars()),original_pixels)
        # LAB-style PLY: RGB plus per-face UVs. Check rendered pixels, rather
        # than only actor visibility, with shading kept on None throughout.
        lab=Path(tmp)/'lab';lab.mkdir()
        source=load_mesh(textured/'geometry.npz')
        source.display.GetPointData().SetScalars(numpy_to_vtk(np.tile([60,120,180],(4,1)).astype(np.uint8),deep=True))
        export_ply(source,np.full(2,-2,np.int8),lab/'roi.ply')
        window.load_folder(lab);assert window.mesh.display.GetNumberOfPoints()==6
        window.shading_picker.setCurrentIndex(0);window.show_overlay.setChecked(False)
        def pixels():
            window.view.GetRenderWindow().Render()
            capture=vtk.vtkWindowToImageFilter();capture.SetInput(window.view.GetRenderWindow())
            capture.ReadFrontBufferOff();capture.SetInputBufferTypeToRGB();capture.Update()
            return vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).copy()
        for shading in range(3):
            window.shading_picker.setCurrentIndex(shading);before=pixels();geometry=window.mesh.fingerprint
            for _ in range(3):
                window.color_toggle.setChecked(False);gray=pixels()
                visible=(gray.min(axis=1)>100)&(np.ptp(gray.astype(np.int16),axis=1)<10)
                assert visible.sum()>1000,('Gray textured mesh disappeared',shading)
                assert window.shading_picker.currentIndex()==shading and bool(window.view.base.GetProperty().GetLighting())==(shading!=0)
                assert window.mesh.fingerprint==geometry
                window.color_toggle.setChecked(True);np.testing.assert_array_equal(pixels(),before)
        # Points and wireframe must be sparse rendered geometry, including LAB
        # textures with color hidden. S/W/P update the same checkbox controls.
        window.shading_picker.setCurrentIndex(0)
        vertices=window.mesh.vertices
        projected=[]
        for vertex in vertices:
            window.view.renderer.SetWorldPoint(*vertex,1);window.view.renderer.WorldToDisplay()
            projected.append(window.view.renderer.GetDisplayPoint()[:2])
        lower=np.floor(np.min(projected,axis=0)-8).astype(int);upper=np.ceil(np.max(projected,axis=0)+8).astype(int)
        def foreground():
            w,h=window.view.GetRenderWindow().GetSize();image=pixels().reshape(h,w,3)
            crop=image[max(0,lower[1]):min(h,upper[1]),max(0,lower[0]):min(w,upper[0])].astype(np.int16)
            return np.any(np.abs(crop-crop[0,0])>20,axis=2).sum()
        for colored in (False,True):
            window.color_toggle.setChecked(colored);window.view.setFocus();app.processEvents()
            surface_count=foreground();assert surface_count>1000
            QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_S)
            assert not window.surface_toggle.isChecked() and not window.view.base.GetVisibility()
            QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_P)
            assert window.points_toggle.isChecked() and window.view.points.GetVisibility()
            assert window.view.points.GetProperty().GetRepresentation()==vtk.VTK_POINTS
            point_count=foreground();assert 0<point_count<surface_count/5,('Points rendered as a surface or disappeared',point_count,surface_count)
            QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_P)
            QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_W)
            assert window.wire_toggle.isChecked() and window.view.wire.GetVisibility()
            assert window.view.wire.GetProperty().GetRepresentation()==vtk.VTK_WIREFRAME
            wire_count=foreground();assert 0<wire_count<surface_count/5,('Wireframe rendered as a surface or disappeared',wire_count,surface_count)
            QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_W)
            QtTest.QTest.keyClick(window.view,QtCore.Qt.Key.Key_S)
            assert window.surface_toggle.isChecked() and window.view.base.GetProperty().GetRepresentation()==vtk.VTK_SURFACE
            assert foreground()==surface_count
        # Color-off also applies immediately to the next loaded textured mesh.
        window.shading_picker.setCurrentIndex(0);window.color_toggle.setChecked(False)
        window.load_folder(textured);gray=pixels()
        assert ((gray.min(axis=1)>100)&(np.ptp(gray.astype(np.int16),axis=1)<10)).sum()>1000
        window.color_toggle.setChecked(True);window.show_overlay.setChecked(True)
        # A corrected top direction is shared by raters and survives reopening.
        normal=window.view.top_normal.copy();window.flip_top()
        other=AnnotationStore(textured,textured/'annotations','other_dentist')
        np.testing.assert_allclose(read_orientation(other,window.mesh),-normal)
        window.load_folder(textured);np.testing.assert_allclose(window.view.top_normal,-normal)
        assert not errors,errors
        window.close();app.processEvents()
        # Consensus review protects independent annotation files from edits/saves.
        args.annotator='consensus';args.review_mode=True
        review=Annotator(args);review.show_error=lambda text:errors.append(text)
        review.show();review.activateWindow();app.setActiveWindow(review);app.processEvents();review.load_folder(root)
        review.rater.setCurrentText(original_rater);assert review.is_read_only()
        independent=review.store.path_for(review.mesh.path);before=independent.read_bytes();labels=review.labels.copy()
        review.edit(np.array([0]),1);review.save()
        np.testing.assert_array_equal(review.labels,labels);assert independent.read_bytes()==before
        review.rater.setCurrentText('consensus');assert not review.is_read_only()
        review.edit(np.array([0]),1);review.close();app.processEvents()
        assert not errors,errors
    print('PASS: S/P/W rendered modes and checkbox synchronization, A/D saved navigation and boundaries, sorting, RGB/UV preservation, rotation, brush, undo/redo, orientation and consensus')

if __name__=='__main__':run()
