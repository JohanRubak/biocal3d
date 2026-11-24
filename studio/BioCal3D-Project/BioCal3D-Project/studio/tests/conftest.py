import os,sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"app"))

# Hardware picking requires an X11/OpenGL display; other UI tests disable rendering.
def pytest_collection_modifyitems(config, items):
    if not os.environ.get('DISPLAY'):
        import pytest
        for item in items:
            if item.path.name=='test_brush.py':item.add_marker(pytest.mark.skip(reason='Requires an X11/OpenGL display'))
