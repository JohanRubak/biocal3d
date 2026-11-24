# Run on Windows from the package root. A Windows binary cannot be built on Linux.
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules
root=Path(SPECPATH).parent
hidden=collect_submodules('vtkmodules')+['matplotlib.backends.backend_qtagg','PIL.Image','plyfile']
a=Analysis([str(root/'launch_studio.py')],pathex=[str(root/'app')],
    binaries=[],datas=[(str(root/'app'),'app'),(str(root/'assets'),'assets')],
    hiddenimports=hidden,hookspath=[],runtime_hooks=[],excludes=['tkinter','PyQt5','PyQt6','PySide2'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='BioCal3D Studio',debug=False,
    strip=False,upx=False,console=False,icon=str(root/'assets'/'biocal3d_studio.ico'))
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='BioCal3D Studio')
