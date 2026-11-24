"""Check exact-path cleanup, complete scan protection and the revision lease."""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_datasets import VARIANTS, expected_keys
from cleanup_dataset import cleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.revision = 'a'*40
        self.rows = []
        self.remote = {}
        for variant in VARIANTS:
            for scanner, sample in sorted(expected_keys()):
                file = sample + '.ply' if variant in VARIANTS[:2] else sample + '/geometry.npz'
                name = f'{variant}/{scanner}/G{sample[3]}/{file}'
                sha = 'b'*64
                self.rows.append(dict(path=name, expected_bytes='100', expected_source_sha256=sha if variant in VARIANTS[:2] else '',
                                      remote_sha256=sha, status='verified', revision=self.revision))
                self.remote[name] = SimpleNamespace(rfilename=name, size=100, lfs=SimpleNamespace(sha256=sha))
        for name in ('README.md', 'dataset_inventory.csv'):
            self.rows.append(dict(path=name, expected_bytes='100', expected_source_sha256='', remote_sha256='',
                                  status='verified', revision=self.revision))
            self.remote[name] = SimpleNamespace(rfilename=name, size=100, lfs=None)
        self.remote['.gitattributes'] = SimpleNamespace(rfilename='.gitattributes', size=80, lfs=None)
        self.obsolete = ['raw_native/TRIOS5/G1/old-name.ply', 'full_harmonized/TRIOS3/G3/old-name.ply']
        for name in self.obsolete:
            self.remote[name] = SimpleNamespace(rfilename=name, size=99, lfs=None)
        self.report = self.root / 'verified.csv'
        self.write_report()
        self.list = self.root / 'obsolete.txt'
        self.list.write_text('\n'.join(self.obsolete) + '\n')
        self.commits = []
        self.head = self.revision
        def dataset_info(*args, **kwargs):
            return SimpleNamespace(private=True, sha=self.head, siblings=list(self.remote.values()))
        def create_commit(**kwargs):
            self.assertEqual(kwargs['parent_commit'], self.revision)
            self.assertEqual(kwargs['revision'], 'main')
            self.commits.append(kwargs)
            for op in kwargs['operations']:
                self.assertFalse(op.is_folder)
                del self.remote[op.path_in_repo]
            self.head = 'c'*40
            return SimpleNamespace(oid=self.head)
        self.api = SimpleNamespace(dataset_info=dataset_info, create_commit=create_commit)

    def tearDown(self):
        self.temp.cleanup()

    def write_report(self):
        with self.report.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)

    def run_cleanup(self, apply=False):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, {'huggingface_hub':
                SimpleNamespace(CommitOperationDelete=lambda **kwargs: SimpleNamespace(**kwargs))}):
            return cleanup(self.api, 'test/private', self.report, self.list, self.revision, apply=apply)

    def test_preview_does_not_modify_remote(self):
        before = set(self.remote)
        self.assertIsNone(self.run_cleanup())
        self.assertEqual(set(self.remote), before)
        self.assertFalse(self.commits)

    def test_apply_deletes_only_listed_files_and_preserves_all_four_variants(self):
        before = set(self.remote)
        self.assertEqual(self.run_cleanup(True), 'c'*40)
        self.assertEqual(set(self.remote), before - set(self.obsolete))
        self.assertIn('.gitattributes', self.remote)
        self.assertEqual(len(self.commits), 1)
        self.assertTrue(list(self.root.glob('cleanup-completed-*.json')))

    def test_changed_revision_blocks_cleanup(self):
        self.head = 'd'*40
        with self.assertRaisesRegex(ValueError, 'changed since verification'):
            self.run_cleanup(True)
        self.assertFalse(self.commits)

    def test_protected_current_file_and_nondata_path_cannot_be_deleted(self):
        for name in (self.rows[0]['path'], 'README.md', 'raw_native/../README.md'):
            self.list.write_text(name + '\n')
            with self.assertRaises(ValueError):
                self.run_cleanup(True)
        self.assertFalse(self.commits)

    def test_incomplete_study_report_blocks_cleanup(self):
        self.rows.pop(0)
        self.write_report()
        with self.assertRaisesRegex(ValueError, '512 unique'):
            self.run_cleanup(True)
        self.assertFalse(self.commits)

    def test_changed_protected_content_blocks_cleanup(self):
        entry = self.remote[self.rows[0]['path']]
        entry.size = 99
        with self.assertRaisesRegex(ValueError, 'different size'):
            self.run_cleanup(True)
        entry.size = 100
        entry.lfs.sha256 = 'd'*64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_cleanup(True)
        self.assertFalse(self.commits)

    def test_inexact_obsolete_list_blocks_cleanup(self):
        self.list.write_text(self.obsolete[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            self.run_cleanup(True)
        self.assertFalse(self.commits)


if __name__ == '__main__':
    unittest.main()
