"""Mock remote failures and successful completion, without private credentials."""
import contextlib
import csv
import fnmatch
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hub_transfer import binary_attributes, check_remote, configure_binary_storage, update_readme, upload_study_archive
import upload_dataset as uploader
from audit_datasets import VARIANTS, expected_keys
from normalize_data_layout import csv_bytes


class HubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sizes = {v + '/BCG3T1-1.ply': 12 for v in VARIANTS}
        self.rows = [{v: 'BCG3T1-1.ply' for v in VARIANTS} | {'raw_sha256': 'a'*64, 'harmonized_sha256': 'b'*64}]
        self.entries = [SimpleNamespace(rfilename=name, size=size,
                                       lfs=SimpleNamespace(sha256=('a' if name.startswith(VARIANTS[0] + '/') else 'b')*64))
                        for name, size in self.sizes.items()]
        self.api = SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(private=True, sha='revision', siblings=self.entries))

    def tearDown(self):
        self.temp.cleanup()

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return check_remote(self.api, 'test/private', self.sizes, self.rows, self.root)

    def test_explicit_binary_rules_cover_ply_vtp_npz_and_uppercase_and_keep_existing_rules(self):
        attributes = binary_attributes('*.custom filter=lfs diff=lfs merge=lfs -text\n')
        self.assertIn('*.custom', attributes)
        self.assertEqual(attributes, binary_attributes(attributes))
        patterns = [line.split()[0] for line in attributes.splitlines() if 'filter=lfs' in line]
        for filename in ('scan.ply', 'scan.PLY', 'roi.vtp', 'geometry.npz', 'maps.NPZ', 'texture.png'):
            self.assertTrue(any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns), filename)

    def test_success_reports_all_folders_and_retains_obsolete_paths(self):
        self.entries.append(SimpleNamespace(rfilename='raw_native/old.ply', size=1, lfs=None))
        revision, stale = self.check()
        self.assertEqual(revision, 'revision')
        self.assertEqual(stale, ['raw_native/old.ply'])
        with next(self.root.glob('upload-verification-*.csv')).open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row['status'] == 'verified' for row in rows))

    def test_missing_fourth_folder_is_failure_not_success(self):
        self.entries.pop()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.check()

    def test_wrong_size_and_same_size_wrong_source_hash_are_failures(self):
        self.entries[0].size = 1
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            self.check()
        self.entries[0].size = 12
        self.entries[0].lfs.sha256 = 'c'*64
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.check()

    def test_model_must_have_binary_storage_checksum_metadata(self):
        self.entries[0].lfs = None
        with self.assertRaisesRegex(ValueError, 'checksum unavailable'):
            self.check()

    def test_uploader_configures_binary_storage_then_uploads_all_four_roots(self):
        files = {}
        records = []
        for scanner, sample in sorted(expected_keys()):
            row = dict(scanner=scanner, sample=sample, issues='', raw_sha256='a'*64, harmonized_sha256='b'*64)
            row.update({v: scanner + '/' + sample + '.ply' for v in VARIANTS})
            records.append(row)
        for v in VARIANTS:
            path = self.root / v / records[0][v]
            path.parent.mkdir(parents=True)
            path.write_bytes(b'fixture')
            files[v] = [path]
        inventory = self.root / 'audit.csv'
        inventory.write_bytes(csv_bytes(records, list(records[0])))
        events = []
        api = SimpleNamespace(create_repo=lambda *a, **k: None,
                              dataset_info=lambda *a, **k: SimpleNamespace(private=True),
                              list_repo_files=lambda *a, **k: [],
                              upload_file=lambda **k: events.append(('file', k['path_in_repo'])),
                              upload_folder=lambda **k: events.append(('folder', k['path_in_repo'], k['allow_patterns'])))
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=lambda: api)}), \
             patch.object(uploader, 'release_files', side_effect=lambda root, *a, **k: files[root.name]), \
             patch('hub_transfer.check_remote', return_value=('revision', [])), contextlib.redirect_stdout(io.StringIO()):
            uploader.upload('test/private', self.root, audit_inventory=inventory)
        self.assertEqual(events[0], ('file', '.gitattributes'))
        self.assertEqual([event[1] for event in events if event[0] == 'folder'], list(VARIANTS))
        self.assertTrue(all(not event[2][0].startswith(event[1] + '/') for event in events if event[0] == 'folder'))

    def test_readme_only_updates_exact_document_and_protects_other_files(self):
        card = self.root / 'card.md'; card.write_text('# Incoming researchers\n')
        data = SimpleNamespace(rfilename='raw_native/test.ply', size=12, blob_id='blob', lfs=None)
        calls = []
        api = SimpleNamespace(
            dataset_info=lambda *a, **k: SimpleNamespace(private=True, sha='before', siblings=[data]),
            upload_file=lambda **k: calls.append(k) or SimpleNamespace(oid='after'),
            hf_hub_download=lambda **k: str(card))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(update_readme(api, 'test/private', card), 'after')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['path_in_repo'], 'README.md')
        self.assertEqual(calls[0]['parent_commit'], 'before')

    def test_readme_only_cli_needs_no_data_drive_or_audit(self):
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=lambda: 'api')}), \
             patch.object(sys, 'argv', ['upload_dataset.py', '--repo-id', 'test/private', '--readme-only']), \
             patch.object(Path, 'home', return_value=self.root), \
             patch('hub_transfer.update_readme') as update, \
             patch.object(uploader, 'release_files', side_effect=AssertionError('no data selection')), \
             contextlib.redirect_stdout(io.StringIO()):
            uploader.main()
        update.assert_called_once()
        self.assertEqual(update.call_args.args[:2], ('api', 'test/private'))

    def test_readme_only_rejects_public_dataset_before_any_write(self):
        api = SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(private=False))
        with self.assertRaisesRegex(ValueError, 'private'):
            update_readme(api, 'test/public', self.root / 'unused.md')

    def test_readme_only_detects_changed_data_metadata(self):
        card = self.root / 'card.md'; card.write_text('# Card\n')
        data = SimpleNamespace(rfilename='raw_native/test.ply', size=12, blob_id='blob', lfs=None)
        def upload(**kwargs):
            data.size = 13
            return SimpleNamespace(oid='after')
        api = SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(private=True, sha='before', siblings=[data]),
                              upload_file=upload)
        with self.assertRaisesRegex(ValueError, 'non-README'):
            update_readme(api, 'test/private', card)

    def test_study_archive_rejects_wrong_bytes_before_any_remote_call(self):
        import json
        archive = self.root / 'run.zip';archive.write_bytes(b'changed')
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(dict(archive_bytes=7, archive_sha256='a'*64)))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            upload_study_archive(None, 'test/private', archive, manifest)

    def test_study_archive_is_separate_and_protects_existing_scan_files(self):
        import json, hashlib
        content=b'known run archive'
        archive=self.root/'run.zip';archive.write_bytes(content)
        record=dict(archive_bytes=len(content),archive_sha256=hashlib.sha256(content).hexdigest(),
                    artifact_path='artifacts/classification/test/run.zip',manifest_path='artifacts/classification/test/manifest.json')
        manifest=self.root/'manifest.json';manifest.write_text(json.dumps(record))
        current={'raw_native/test.ply':SimpleNamespace(rfilename='raw_native/test.ply',size=12,blob_id='mesh',lfs=None)}
        writes=[]
        def upload(**kwargs):
            writes.append(kwargs)
            name=kwargs['path_in_repo'];data=Path(kwargs['path_or_fileobj']).read_bytes() if isinstance(kwargs['path_or_fileobj'],str) else kwargs['path_or_fileobj']
            current[name]=SimpleNamespace(rfilename=name,size=len(data),blob_id='blob',lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
            return SimpleNamespace(oid='revision')
        api=SimpleNamespace(dataset_info=lambda *a,**k:SimpleNamespace(private=True,sha='revision',siblings=list(current.values())),
                            upload_file=upload,hf_hub_download=lambda **k:str(manifest))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(upload_study_archive(api,'test/private',archive,manifest),'revision')
        self.assertEqual([entry['path_in_repo'] for entry in writes],['.gitattributes',record['artifact_path'],record['manifest_path']])
        self.assertEqual(current['raw_native/test.ply'].blob_id,'mesh')


if __name__ == '__main__':
    unittest.main()
