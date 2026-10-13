"""Portable subset integration checks: relocation, colors, face IDs and labels."""
import contextlib
import csv
import io
import json
import shutil
import unittest
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image
from annotator.mesh_io import AnnotationStore,load_mesh,discover,export_ply,polydata,Mesh
from analyse_labeling_agreement import analyse
from prepare_pilot_study import prepare,write_csv,digest
from prepare_shared_study import export_shared,copy_scan
import test_pilot_study as fixtures


class SharedStudyTests(fixtures.PilotTests):
    # Use the 512-scan fixture, without rerunning inherited test cases here.
    test_default_is_g5_only_two_calibration_six_agreement=None
    test_explicit_ids_validation_and_insufficient_scanner_coverage=None
    test_full_coverage_named_roles_partial_overlap_and_no_consensus=None

    def quiet_export(self,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return export_shared(self.root,**kwargs)

    def test_existing_selection_portable_inputs_and_moved_analysis(self):
        with contextlib.redirect_stdout(io.StringIO()):_,old=prepare(self.root,app_launcher=self.launcher)
        # Existing labels are deliberately present in the source; never copy them.
        marker=self.root/'annotations'/'bruna'/'existing.labels.npz';marker.write_bytes(b'not a study input')
        output=self.root.parent/(self.root.name+'_shared')
        try:
            config=self.quiet_export(output=output)
            self.assertEqual(config['calibration_ids'],old['calibration_ids'])
            self.assertEqual(config['agreement_ids'],old['agreement_ids'])
            self.assertTrue(json.loads((output/'study_plan.json').read_text())['reused_existing_selection'])
            self.assertEqual(len(discover(output/'calibration')),24)
            self.assertEqual(len(discover(output/'agreement')),72)
            self.assertFalse(list(output.rglob('*.bat')))
            self.assertFalse(list(output.rglob('*.labels.npz')))
            for phase in ('calibration','agreement'):
                for name in ('bruna','katrin','johan'):self.assertTrue((output/phase/'annotations'/name).is_dir())
                for path in discover(output/phase):
                    metadata=json.loads(path.with_name('metadata.json').read_text())
                    self.assertIn(metadata['stage'],('baseline','partial','clean'))
                    self.assertTrue(metadata['portable_dataset'])
            with self.assertRaises(ValueError):self.quiet_export(output=output)
            # Mimic moving to a drive with a different absolute path.
            moved=output.with_name(output.name+'_moved');shutil.move(output,moved);output=moved
            self.root.joinpath('samples').rename(self.root/'unavailable_source_samples')
            root=moved/'agreement'
            for path in discover(root):
                mesh=load_mesh(path)
                for name in ('bruna','katrin'):
                    AnnotationStore(root,root/'annotations',name).save(mesh,np.array([1,0],np.int8),True,'independent')
                AnnotationStore(root,root/'annotations','johan').save(mesh,np.array([0,1],np.int8),True,'experiment')
            with contextlib.redirect_stdout(io.StringIO()):summary=analyse(root,root)
            self.assertEqual(summary['raters'],['bruna','katrin'])
            self.assertEqual(summary['complete_pairs'],72)
            self.assertEqual(summary['partial_stage_primary']['scans'],24)
            self.assertEqual(summary['partial_stage_primary']['specimens'],6)
            self.assertEqual(summary['partial_stage_primary']['plaque_dice']['mean_of_specimen_means'],1)
            with self.assertRaises(ValueError):analyse(moved/'calibration',moved/'calibration')
        finally:shutil.rmtree(output,ignore_errors=True)

    def test_no_launcher_required_and_originals_untouched(self):
        self.launcher.unlink()
        before={p.relative_to(self.root).as_posix():digest(p) for p in self.root.rglob('*') if p.is_file()}
        output=self.root.parent/(self.root.name+'_shared')
        try:
            config=self.quiet_export(output=output)
            self.assertEqual(config['counts']['calibration_bruna']['scans'],24)
            self.assertEqual(config['counts']['agreement_bruna']['scans'],72)
            after={p.relative_to(self.root).as_posix():digest(p) for p in self.root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
        finally:shutil.rmtree(output,ignore_errors=True)

    def test_texture_and_orientation_survive_without_original_files(self):
        # Test binary PLY and VTP with UVs and absolute external texture hints.
        v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        f=np.array([[0,1,2],[1,3,2]]);uv=v[:,:2]
        raw=self.root/'external_images';raw.mkdir()
        image=raw/'scanner_photo.jpg';Image.new('RGB',(4,4),(140,30,70)).save(image)
        source=self.root/'texture_inputs';source.mkdir()
        ply=source/'source.ply'
        export_ply(Mesh(source/'input.ply',v,f,polydata(v,f,uv=uv),np.ones(2,bool),image),None,ply)
        # Force the source PLY to depend on the external original path.
        from prepare_shared_study import patch_ply_texture
        patched=source/'temporary.ply';patch_ply_texture(ply,patched,str(image));patched.replace(ply)
        (source/'source.texture.jpg').unlink()
        vtp=source/'source.vtp';poly=polydata(v,f,uv=uv)
        field=vtk.vtkStringArray();field.SetName('TextureFile');field.InsertNextValue(str(image));poly.GetFieldData().AddArray(field)
        writer=vtk.vtkXMLPolyDataWriter();writer.SetInputData(poly);writer.SetFileName(str(vtp));writer.Write()
        np.savez_compressed(source/'geometry.npz',vertices=v,faces=f,original_to_canonical=np.eye(4))
        npz=source/'npz_input'/'geometry.npz';npz.parent.mkdir()
        np.savez_compressed(npz,vertices=v,faces=f,uv=uv,texture_path=str(image),original_to_canonical=np.eye(4))
        destination=self.root/'portable_test';destination.mkdir()
        expected={};images={};expected_uv={}
        for path in (ply,vtp,npz):
            folder=destination/path.suffix[1:];folder.mkdir()
            mesh=load_mesh(path)
            row=copy_scan(self.root,folder,dict(annotation_mesh=str(path.relative_to(self.root)),physical_id='G5_S1',scanner='LABscanner',stage='partial'))
            copied=folder/row['annotation_mesh'];expected[copied]=mesh.fingerprint;images[copied]=image.read_bytes()
            expected_uv[copied]=vtk_to_numpy(mesh.display.GetPointData().GetTCoords()).copy()
        # Remove original geometry and photo; portable copies must still work.
        shutil.rmtree(raw);shutil.rmtree(source)
        for path,fingerprint in expected.items():
            mesh=load_mesh(path);self.assertEqual(mesh.fingerprint,fingerprint)
            self.assertEqual(mesh.warning,'');self.assertTrue(mesh.texture.is_relative_to(destination))
            self.assertEqual(mesh.texture.read_bytes(),images[path])
            np.testing.assert_array_equal(vtk_to_numpy(mesh.display.GetPointData().GetTCoords()),expected_uv[path])
            self.assertEqual(discover(path.parents[1]),[path])
            with np.load(path.with_name('geometry.npz'),allow_pickle=False) as data:
                np.testing.assert_array_equal(data['original_to_canonical'],np.eye(4))

    def test_labeled_input_fails_atomically(self):
        row=next(r for r in self.rows if r['group']==5)
        path=self.root/row['annotation_mesh'];mesh=load_mesh(path)
        # Replace every G5 input with labeled geometry, guaranteeing selection hits it.
        for r in self.rows:
            if r['group']!=5:continue
            path=self.root/r['annotation_mesh'];poly=polydata(mesh.vertices,mesh.faces,np.tile([120,45,90],(4,1)))
            labels=vtk.vtkIntArray();labels.SetName('face_labels');labels.InsertNextValue(1);labels.InsertNextValue(0)
            poly.GetCellData().AddArray(labels)
            w=vtk.vtkXMLPolyDataWriter();w.SetInputData(poly);w.SetFileName(str(path));w.Write()
        output=self.root.parent/(self.root.name+'_shared')
        with self.assertRaisesRegex(ValueError,'already contains ROI labels'):self.quiet_export(output=output)
        self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
