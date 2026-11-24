"""BioCal3D Studio entry point. Startup errors are retained in a per-user log."""
from pathlib import Path
import os
import sys
import traceback

ROOT=Path(sys._MEIPASS) if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'app'))

def launch():
    try:
        from qt_bootstrap import prepare_qt, apply_library_paths
        setup=prepare_qt()
        from PySide6 import QtCore
        apply_library_paths(QtCore,setup)
        from studio import main
        return main()
    except Exception:
        message=traceback.format_exc()
        log_root=Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local'/'state'))/'BioCal3DStudio'/'logs'
        log_root.mkdir(parents=True,exist_ok=True);log=log_root/'startup.log';log.write_text(message,encoding='utf-8')
        print(message,file=sys.stderr)
        try:
            from PySide6 import QtWidgets
            app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            QtWidgets.QMessageBox.critical(None,'BioCal3D Studio could not start',f'{message}\n\nLog: {log}')
        except Exception:pass
        return 1

if __name__=='__main__':sys.exit(launch())
