import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_files import seed_reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.refs = Path(__file__).resolve().parents[1] / 'references'

    def tearDown(self):
        self.temp.cleanup()

    def test_all_four_original_fits_copy_exact_bytes_and_preserve_existing_working_fit(self):
        for scanner in ('iTERO', 'LABscanner', 'TRIOS3', 'TRIOS5'):
            with contextlib.redirect_stdout(io.StringIO()):
                target = seed_reference(scanner, self.base, self.refs)
            self.assertEqual(target.read_bytes(), (self.refs / scanner / 'reference_roi.npz').read_bytes())
            target.write_bytes(b'working fit must not be overwritten')
            self.assertEqual(seed_reference(scanner, self.base, self.refs).read_bytes(), b'working fit must not be overwritten')

    def test_modified_reference_is_rejected_by_checksum(self):
        copied = self.base / 'source'; shutil.copytree(self.refs, copied)
        with (copied / 'iTERO/reference_roi.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            seed_reference('iTERO', self.base / 'output', copied)
        self.assertFalse((self.base / 'output/iTERO/reference_roi.npz').exists())

    def test_no_source_keeps_original_manual_reference_workflow(self):
        target = seed_reference('TRIOS5', self.base)
        self.assertFalse(target.exists())

    def test_reference_sample_must_match_the_configured_scan(self):
        with self.assertRaisesRegex(ValueError, 'reference sample'):
            seed_reference('TRIOS5', self.base, self.refs, reference_sample_id='BCG2T0-1')
        self.assertFalse((self.base / 'TRIOS5/reference_roi.npz').exists())


if __name__ == '__main__':unittest.main()
