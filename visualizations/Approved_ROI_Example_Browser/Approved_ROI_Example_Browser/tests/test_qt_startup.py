import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qt_bootstrap
import launch_browser


class QtStartupTests(unittest.TestCase):
    def test_binding_plugins_replace_foreign_paths_and_retain_dll_handles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); binding = root / 'site-packages' / 'PySide6'
            platforms = binding / 'plugins' / 'platforms'; platforms.mkdir(parents=True)
            (platforms / 'qwindows.dll').touch()
            env = {'QT_PLUGIN_PATH': 'foreign/Qt5/plugins', 'QT_QPA_PLATFORM_PLUGIN_PATH': 'opencv/plugins', 'PATH': 'original-path'}
            handles = []
            def add(path):
                handle = object(); handles.append(handle); return handle
            result = qt_bootstrap.configure_windows(binding, root, env, add)
            self.assertEqual(env['QT_PLUGIN_PATH'], str(binding / 'plugins'))
            self.assertEqual(env['QT_QPA_PLATFORM_PLUGIN_PATH'], str(platforms))
            self.assertEqual(env['QT_QPA_PLATFORM'], 'windows')
            self.assertTrue(env['PATH'].endswith('original-path'))
            self.assertEqual(result['replaced_environment']['QT_PLUGIN_PATH'], 'foreign/Qt5/plugins')
            self.assertTrue(all(handle in qt_bootstrap._dll_handles for handle in handles))

    def test_missing_windows_plugin_fails_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            env = {'QT_PLUGIN_PATH': 'original'}
            with self.assertRaisesRegex(RuntimeError, 'qwindows.dll'):
                qt_bootstrap.configure_windows(Path(folder) / 'missing', folder, env, lambda _: None)
            self.assertEqual(env, {'QT_PLUGIN_PATH': 'original'})

    def test_conda_library_plugin_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); binding = root / 'Lib' / 'site-packages' / 'PySide6'; binding.mkdir(parents=True)
            platforms = root / 'Library' / 'plugins' / 'platforms'; platforms.mkdir(parents=True)
            (platforms / 'qwindows.dll').touch()
            result = qt_bootstrap.configure_windows(binding, root, {}, lambda _: object())
            self.assertEqual(result['plugins'], str(platforms.parent))

    def test_failed_probe_logs_debug_retry_without_launching_browser(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'qt_bootstrap.py').write_text("import sys; print('simulated missing DLL', flush=True); sys.exit(3)")
            with patch.object(launch_browser, '__file__', str(root / 'launch_browser.py')), patch.object(sys, 'argv', ['launch_browser.py']):
                self.assertEqual(launch_browser.main(), 1)
            log = (root / 'qt_startup.log').read_text()
            self.assertIn('Detailed retry:', log); self.assertEqual(log.count('simulated missing DLL'), 2)


if __name__ == '__main__': unittest.main()
