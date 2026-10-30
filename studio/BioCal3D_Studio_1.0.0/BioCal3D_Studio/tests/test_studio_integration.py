"""Regression tests for shared workspaces, reviewed inspection, and both zoom paths."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets, QtTest
import vtk
from test_roi_ui import NoRenderWindow, NoRenderInteractor


@pytest.fixture(scope='module')
def app():
    qapp=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return qapp


@pytest.fixture
def no_render(monkeypatch):
    import vtkmodules.qt.QVTKRenderWindowInteractor as q
    monkeypatch.setattr(q,'vtkRenderWindow',NoRenderWindow)
    monkeypatch.setattr(q,'vtkGenericRenderWindowInteractor',NoRenderInteractor)


def wait_until(app, test, timeout=10):
    deadline=time.monotonic()+timeout
    while not test() and time.monotonic()<deadline:app.processEvents();QtTest.QTest.qWait(5)
    assert test(),'Background operation did not complete'


def test_orientation_native_zoom_pan_rotation_and_angle_preserve_linked_cameras(app,no_render):
    from orientation import OrientationPage
    import roi_browser_core as display
    from mesh_io import polydata
    page=OrientationPage();page.show();app.processEvents()
    vertices=np.array([[-2.,-2.,0],[2.,-2.,.1],[2.,2.,0],[-2.,2.,0]])
    model={'poly':polydata(vertices,np.array([[0,1,2],[0,2,3]]),np.tile([120,170,150],(4,1))),
           'association':'point','texture':None,'bounds':[-2,2,-2,2,0,.1],
           'path':'fixture.ply','sample_id':'BCG5T1-1','appearance':'vertex RGB','source_sha256':'fixture'}
    page.reference=model;page.source=model;page.draw();page.fit()
    view=page.views;renderer=view.renderers[1];camera=renderer.GetActiveCamera();before=camera.GetParallelScale()
    interactor=view.widgets[1].GetRenderWindow().GetInteractor();style=interactor.GetInteractorStyle()
    view.widgets[1].GetRenderWindow().SetSize(320,320);interactor.SetEventInformation(80,80);style.SetCurrentRenderer(renderer)
    style.OnMouseWheelForward();app.processEvents()
    assert camera.GetParallelScale()<before
    for r in view.renderers:assert display.camera_state(r)==display.camera_state(renderer)
    # The same native middle-button trackball path used by the inspection view.
    old_focus=camera.GetFocalPoint();style.OnMiddleButtonDown();interactor.SetEventInformation(105,92);style.OnMouseMove();style.OnMiddleButtonUp();app.processEvents()
    assert camera.GetFocalPoint()!=old_focus
    for r in view.renderers:assert display.camera_state(r)==display.camera_state(renderer)
    camera.Azimuth(20);app.processEvents();expected=display.camera_state(renderer)
    page.angle.setValue(42.5)
    for r in view.renderers:assert display.camera_state(r)==expected
    page.landmarks();assert page.views.pick_mode
    page.pick_landmark(0,np.array([0.,0,0]));page.pick_landmark(0,np.array([2.,0,0]))
    page.pick_landmark(1,np.array([0.,0,0]));page.pick_landmark(1,np.array([0.,2,0]))
    assert page.angle.value()==pytest.approx(-47.5)
    assert not page.views.pick_mode
    page.close()


def test_browser_camera_modification_and_wheel_sync_all_twelve_views(app,no_render,tmp_path):
    from inspection import create_browser
    import roi_browser_core as core
    from test_browser_core import sample
    path=sample(tmp_path)
    browser=create_browser(argparse.Namespace(root=None,output=None,pixels=400,auto_restore=False))
    mesh=core.load_mesh(core.Scan('iTERO',3,0,1,path),tmp_path)
    for view in browser.views:view.set_mesh(mesh)
    browser.set_ready(True);browser.reset_top();app.processEvents()
    source=browser.views[4];camera=source.renderer.GetActiveCamera();before=camera.GetParallelScale()
    camera.SetParallelScale(before*.7)
    app.processEvents()
    for view in browser.views:assert view.renderer.GetActiveCamera().GetParallelScale()==pytest.approx(before*.7)
    # Exercise the VTK trackball style's actual mouse-wheel zoom path.
    interactor=source.widget.GetRenderWindow().GetInteractor();style=interactor.GetInteractorStyle()
    style.SetCurrentRenderer(source.renderer);style.OnMouseWheelForward();app.processEvents()
    for view in browser.views:assert core.camera_state(view.renderer)==core.camera_state(source.renderer)
    app.processEvents();assert not browser.sync_timer.isActive()
    browser.close()


def test_inspection_uses_current_review_and_does_not_write(app,tmp_path):
    from test_orientation_engine import make_fixture
    import roi_browser_core as core
    from mesh_io import AnnotationStore,load_mesh
    from roi_review_store import read_reviewed,save_review,load_review_context
    make_fixture(tmp_path/'dataset');root=tmp_path/'dataset'
    path=root/'samples'/'TRIOS5'/'G5'/'BCG5T1-1'/'roi.ply'
    store=AnnotationStore(root,root/'annotations','test')
    mesh=read_reviewed(store,path)
    base,mask,settings,_,raw=load_review_context(store,mesh)
    mask[:]=False;mask[0]=True
    save_review(store,path,base,mask,settings,'approved','integration fixture',mesh.review_version,raw)
    before={str(p.relative_to(root)):(p.stat().st_mtime_ns,p.stat().st_size) for p in root.rglob('*') if p.is_file()}
    scan=core.Scan('TRIOS5',5,1,1,path.with_name('geometry.npz'))
    display=core.load_mesh(scan,root)
    assert display['faces']==1 and display['review_status']=='approved'
    after={str(p.relative_to(root)):(p.stat().st_mtime_ns,p.stat().st_size) for p in root.rglob('*') if p.is_file()}
    assert before==after


def test_studio_switches_workspaces_saves_labels_and_scopes_space(app,no_render,tmp_path,monkeypatch):
    from studio import Studio
    from test_orientation_engine import make_fixture
    from roi_review_store import read_reviewed
    from mesh_io import AnnotationStore
    root=tmp_path/'dataset';config,_=make_fixture(root)
    args=argparse.Namespace(root=root,output=None,annotator='studio_test',manifest=None,lock_annotator=False,review_mode=False,no_local_cache=True,cache_dir=None,page='label')
    w=Studio(args);w.show();app.processEvents()
    label=w.pages[3];assert label.root==root;assert len(label.queue)==4
    # Explicit edits must be retained when opening another workspace.
    label.labels[:]=0;label.dirty=True;edited=label.mesh.path
    original_save=label.save
    monkeypatch.setattr(label,'save',lambda **kwargs:False)
    w.nav.setCurrentRow(1);app.processEvents()
    assert w.active_page==3 and w.nav.currentRow()==3 and 1 not in w.pages
    monkeypatch.setattr(label,'save',original_save)
    w.nav.setCurrentRow(1);app.processEvents();orientation=w.pages[1]
    wait_until(app,lambda:not orientation.busy)
    assert orientation.data_root==root
    saved=AnnotationStore(root,root/'annotations','studio_test').path_for(edited)
    with np.load(saved,allow_pickle=False) as z:assert np.all(z['face_labels']==0)
    # A hidden labeling viewport must not steal Space in the orientation page.
    event=QtCore.QEvent(QtCore.QEvent.Type.KeyPress)
    assert not label.view.rotation_keys.eligible(orientation)
    w.nav.setCurrentRow(0);app.processEvents();browser=w.pages[0]
    wait_until(app,lambda:not browser.tasks)
    assert browser.root==root;assert w.stack.count()==3
    w.nav.setCurrentRow(3);app.processEvents();assert w.pages[3] is label
    assert not label.dirty
    # Applying orientation must work even if labeling was opened earlier.
    w.nav.setCurrentRow(1);app.processEvents();assert w.prepare_orientation()
    assert 3 not in w.pages and not list(root.rglob('*.lock'))
    import orient_scanners
    output=tmp_path/'oriented'
    orient_scanners.apply_tree(root,output,config,lambda _:None)
    assert (output/'scanner_orientation.json').is_file()
    w.close();app.processEvents();assert not w.isVisible()


def test_dedicated_review_saves_roi_and_migrates_open_labels(app,no_render,tmp_path,monkeypatch):
    from studio import Studio
    from test_orientation_engine import make_fixture
    from roi_review_store import read_state
    from PySide6 import QtWidgets
    root=tmp_path/'dataset';make_fixture(root)
    args=argparse.Namespace(root=root,output=None,annotator='studio_review_test',manifest=None,lock_annotator=False,review_mode=False,no_local_cache=True,cache_dir=None,page='label')
    w=Studio(args);w.show();app.processEvents();label=w.pages[3]
    assert not hasattr(label,'roi_review_button')
    label.labels[:]=[1,0,1,0];label.dirty=True;path=label.mesh.path
    w.nav.setCurrentRow(2);app.processEvents();review=w.pages[2]
    assert review.mesh.path==path and review.editor.embedded and not review.editor.isWindow()
    assert [w.nav.item(i).text() for i in range(4)]==['Inspect','Orient','Review','Label']
    editor=review.editor;editor.live.setChecked(False);editor.operation.setCurrentIndex(0)
    editor.start_stroke();editor.paint_faces([0]);editor.end_stroke();assert editor.has_changes()
    # Cancel a switch with unsaved corrections; no workspace change is allowed.
    monkeypatch.setattr(QtWidgets.QMessageBox,'question',lambda *_,**__:QtWidgets.QMessageBox.StandardButton.Cancel)
    w.nav.setCurrentRow(3);assert w.active_page==2
    assert editor.commit('approved');app.processEvents()
    assert editor.isVisible() and not editor.has_changes()
    assert read_state(review.store,path)['status']=='approved'
    assert len(label.labels)==3;np.testing.assert_array_equal(label.labels,[0,1,0])
    w.nav.setCurrentRow(3);app.processEvents();assert w.pages[3] is label and label.mesh.path==path
    assert not label.dirty
    w.close();app.processEvents()


def test_review_alone_does_not_create_or_modify_plaque_labels(app,no_render,tmp_path):
    import hashlib
    from review_workspace import ReviewPage
    from test_orientation_engine import make_fixture
    root=tmp_path/'dataset';make_fixture(root)
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.labels.npz')}
    page=ReviewPage(argparse.Namespace(output=None,manifest=None,no_local_cache=True,cache_dir=None))
    page.show();assert page.load_folder(root)
    page.editor.notes.setPlainText('Dedicated ROI review');assert page.editor.commit('flagged')
    after={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.labels.npz')}
    assert before==after and not (root/'annotations'/'roi_review').exists()
    page.close()


def test_orientation_original_geometry_enters_verified_frame(tmp_path):
    from test_orientation_engine import make_fixture
    from orientation_preview import preview_model
    from original_models import OriginalModelIndex
    from mesh_io import load_mesh,Mesh,polydata,export_ply,atomic_npz
    import orient_scanners as core
    from vtk.util.numpy_support import vtk_to_numpy
    root=tmp_path/'dataset';_,rows=make_fixture(root);row=rows[0]
    folder=root/row['sample_dir'];g=core.arrays(folder/'geometry.npz')
    original=load_mesh(folder/'roi.ply');t=g['original_to_canonical']
    raw=tmp_path/'raw'/row['scanner']/'G5'/row['sample_id']/'source.ply';raw.parent.mkdir(parents=True)
    vertices=core.points(original.vertices,np.linalg.inv(t));colors=np.tile([110,170,145],(len(vertices),1))
    export_ply(Mesh(raw.with_name('input.ply'),vertices,original.faces,polydata(vertices,original.faces,colors),np.ones(len(original.faces),bool)),None,raw)
    g['source_ply']=np.array(str(raw));g['source_sha256']=np.array(core.sha(raw));atomic_npz(folder/'geometry.npz',**g)
    model=preview_model(root,row,OriginalModelIndex(tmp_path/'raw'))
    np.testing.assert_allclose(vtk_to_numpy(model['poly'].GetPoints().GetData()),original.vertices,atol=1e-6)
    assert model['poly'].GetNumberOfCells()==len(original.faces)
    assert model['source_sha256']==core.sha(raw)


def test_label_opened_after_review_persists_migrated_label_version(app,no_render,tmp_path):
    from test_orientation_engine import make_fixture
    from review_workspace import ReviewPage
    from biocal3d_annotator import Annotator
    root=tmp_path/'dataset';make_fixture(root)
    # The orientation-engine fixture omits camera scale; add a complete UI view.
    import json
    from mesh_io import atomic_npz
    for path in (root/'annotations'/'bruna').rglob('*.labels.npz'):
        with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        camera=json.loads(str(data['camera'].item()));camera['parallel_scale']=7.
        data['camera']=np.array(json.dumps(camera));atomic_npz(path,**data)
    page=ReviewPage(argparse.Namespace(output=None,manifest=None,no_local_cache=True,cache_dir=None));page.load_folder(root)
    edited=page.mesh.path;page.editor.live.setChecked(False);page.editor.operation.setCurrentIndex(0)
    page.editor.start_stroke();page.editor.paint_faces([0]);page.editor.end_stroke();assert page.editor.commit('approved')
    version=page.editor.version;page.close()
    label=Annotator(argparse.Namespace(root=None,output=None,annotator='bruna',demo=False,no_local_cache=True))
    errors=[];label.show_error=errors.append
    label.load_folder(root);assert not errors,errors
    label.list.setCurrentRow(label.queue.index(edited))
    assert len(label.labels)==3 and label.dirty
    assert label.save(only_if_dirty=True)
    with np.load(label.store.path_for(edited),allow_pickle=False) as z:
        assert str(z['geometry_review_version'].item())==version and len(z['face_labels'])==3
    label.close()
