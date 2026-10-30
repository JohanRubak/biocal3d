"""Keep this application's Windows Qt paths tied to its selected PySide6."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys

_dll_handles = []


def configure_windows(binding_dir, prefix, environ, add_dll_directory):
    binding_dir, prefix = Path(binding_dir), Path(prefix)
    plugin_candidates = [binding_dir / 'plugins', binding_dir / 'Qt' / 'plugins', prefix / 'Library' / 'plugins']
    plugins = next((p for p in plugin_candidates if (p / 'platforms' / 'qwindows.dll').is_file()), None)
    if plugins is None:
        raise RuntimeError('The selected PySide6 installation has no qwindows.dll. Run Diagnose_Qt.bat to record the environment before changing packages.')
    removed = {}
    for key in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QML2_IMPORT_PATH', 'QML_IMPORT_PATH'):
        if key in environ: removed[key] = environ.pop(key)
    environ['QT_PLUGIN_PATH'] = str(plugins)
    environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins / 'platforms')
    environ['QT_API'] = 'pyside6'
    environ['QT_QPA_PLATFORM'] = 'windows'
    directories = [p for p in (binding_dir, binding_dir / 'Qt' / 'bin', binding_dir / 'Qt6' / 'bin', binding_dir.parent / 'shiboken6') if p.is_dir()]
    for directory in directories: _dll_handles.append(add_dll_directory(str(directory)))
    environ['PATH'] = os.pathsep.join([*(str(p) for p in directories), environ.get('PATH', '')])
    return {'binding_directory': str(binding_dir), 'plugins': str(plugins), 'replaced_environment': removed}


def prepare_qt():
    if sys.platform != 'win32': return None
    spec = importlib.util.find_spec('PySide6')
    if spec is None or spec.origin is None: raise RuntimeError('PySide6 is not installed in ' + sys.executable)
    return configure_windows(Path(spec.origin).parent, sys.prefix, os.environ, os.add_dll_directory)


def apply_library_paths(QtCore, setup):
    if setup: QtCore.QCoreApplication.setLibraryPaths([setup['plugins']])


if __name__ == '__main__':
    import importlib.metadata
    import json
    print('Python:', sys.executable, flush=True)
    print('Prefix:', sys.prefix, flush=True)
    for name in ('PySide6', 'PySide6-Essentials', 'PySide6-Addons', 'shiboken6'):
        try: print(name + ':', importlib.metadata.version(name), flush=True)
        except importlib.metadata.PackageNotFoundError: print(name + ': not separately installed', flush=True)
    setup = prepare_qt()
    print('Qt path setup:', json.dumps(setup), flush=True)
    from PySide6 import QtCore, QtWidgets
    apply_library_paths(QtCore, setup)
    print('Qt runtime:', QtCore.qVersion(), flush=True)
    print('Qt plugin search:', QtCore.QCoreApplication.libraryPaths(), flush=True)
    app = QtWidgets.QApplication([])
    print('Qt platform:', app.platformName(), flush=True)
    print('QApplication startup passed', flush=True)
    app.quit()
