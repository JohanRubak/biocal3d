"""One shared desktop workspace for the BioCal3D dataset."""
from pathlib import Path
import argparse
import hashlib
import sys
from PySide6 import QtCore, QtGui, QtWidgets

APP_ROOT = Path(sys._MEIPASS) if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[1]


class Studio(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__();from theme import apply_theme
        apply_theme(self)
        self.args=args;self.pages={};self.surfaces={};self.root=None;self.raw_root=None
        self.settings=QtCore.QSettings('BioCal3D','Studio');self.setWindowTitle('BioCal3D Studio');self.resize(1600,1000)
        self.setWindowIcon(QtGui.QIcon(str(APP_ROOT/'assets'/'biocal3d_studio.svg')))
        central=QtWidgets.QWidget();self.setCentralWidget(central);layout=QtWidgets.QHBoxLayout(central);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        sidebar=QtWidgets.QFrame();sidebar.setObjectName('studioSidebar');sidebar.setFixedWidth(175);side=QtWidgets.QVBoxLayout(sidebar);side.setContentsMargins(18,24,18,20);layout.addWidget(sidebar)
        logo=QtWidgets.QLabel();logo.setPixmap(self.windowIcon().pixmap(70,70));side.addWidget(logo)
        brand=QtWidgets.QLabel('BioCal3D\nStudio');brand.setStyleSheet('font-size:23px;font-weight:700;color:#edf4ff');side.addWidget(brand)
        caption=QtWidgets.QLabel('DATASET WORKSPACE');caption.setStyleSheet('color:#9daec4;font-size:10px;letter-spacing:1px');side.addWidget(caption);side.addSpacing(25)
        self.nav=QtWidgets.QListWidget();self.nav.setObjectName('studioNav');self.nav.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for title in ['Inspect','Orient','Review','Label']:self.nav.addItem(title)
        self.nav.setMaximumHeight(230);side.addWidget(self.nav)
        b=QtWidgets.QPushButton('Open dataset…');b.clicked.connect(self.choose_dataset);side.addWidget(b)
        b=QtWidgets.QPushButton('Original models…');b.clicked.connect(self.choose_raw);side.addWidget(b)
        self.dataset_label=QtWidgets.QLabel('No dataset selected');self.dataset_label.setWordWrap(True);self.dataset_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse);side.addWidget(self.dataset_label)
        self.output_button=QtWidgets.QPushButton('Open oriented copy');self.output_button.hide();self.output_button.clicked.connect(self.use_output);side.addWidget(self.output_button)
        side.addStretch();help_button=QtWidgets.QPushButton('Quick guide');help_button.clicked.connect(self.guide);side.addWidget(help_button)
        version=QtWidgets.QLabel('v1.0.0');version.setStyleSheet('color:#9daec4');side.addWidget(version)
        self.stack=QtWidgets.QStackedWidget();layout.addWidget(self.stack,1)
        self.nav.currentRowChanged.connect(self.switch_page)
        self.root=Path(args.root).resolve() if args.root else None
        # A remembered path is shown as a convenience; do not silently open a shared study.
        if self.root is None:
            remembered=self.settings.value('dataset','')
            if remembered and Path(remembered).is_dir():self.root=Path(remembered)
        self.restore_raw();self.update_label();self.nav.setCurrentRow({'inspect':0,'orient':1,'review':2,'label':3}[args.page])
        self.restoreGeometry(self.settings.value('geometry',QtCore.QByteArray()))
    def update_label(self):
        self.dataset_label.setText(str(self.root) if self.root else 'No dataset selected')
        self.dataset_label.setToolTip('Active dataset\n'+str(self.root or ''))
    def restore_raw(self):
        key='originals/'+hashlib.sha256(str(self.root).encode()).hexdigest()[:20]
        raw=self.settings.value(key,'');self.raw_root=Path(raw) if raw and Path(raw).is_dir() else None
    def remember_dataset(self, root, origin):
        root=Path(root).resolve()
        if root!=self.root:
            self.root=root;self.settings.setValue('dataset',str(root));self.restore_raw();self.update_label()
        self.apply_raw(origin)
    def page_args(self):
        return argparse.Namespace(root=None,output=self.args.output,annotator=self.args.annotator,manifest=self.args.manifest,
             lock_annotator=self.args.lock_annotator,review_mode=self.args.review_mode,no_local_cache=self.args.no_local_cache,
             cache_dir=self.args.cache_dir,demo=False,pixels=1600,auto_restore=False)
    def create_page(self, number):
        if number==0:
            from inspection import create_browser
            page=create_browser(self.page_args());page.setWindowFlags(QtCore.Qt.WindowType.Widget)
        elif number==1:
            from orientation import OrientationPage
            page=OrientationPage();page.before_apply=self.prepare_orientation;page.output_ready.connect(self.output_ready)
        elif number==2:
            from review_workspace import ReviewPage
            page=ReviewPage(self.page_args());page.surface_changed.connect(self.review_surface_changed)
        else:
            from biocal3d_annotator import Annotator
            page=Annotator(self.page_args());page.setWindowFlags(QtCore.Qt.WindowType.Widget)
        page.dataset_changed.connect(lambda root,n=number:self.remember_dataset(root,n))
        if hasattr(page,'originals_changed'):page.originals_changed.connect(self.remember_raw)
        self.pages[number]=page
        surface=QtWidgets.QScrollArea();surface.setFrameShape(QtWidgets.QFrame.Shape.NoFrame);surface.setWidgetResizable(True);surface.setWidget(page)
        self.surfaces[number]=surface;self.stack.addWidget(surface);return page
    def switch_page(self, number):
        previous=getattr(self,'active_page',None)
        def stay():
            self.nav.blockSignals(True);self.nav.setCurrentRow(previous);self.nav.blockSignals(False)
        orientation=self.pages.get(1)
        if number in (2,3) and orientation and orientation.applying:
            self.statusBar().showMessage('Wait for the oriented copy to finish before editing surfaces or labels.');stay();return
        review=self.pages.get(2);label=self.pages.get(3)
        if previous==2 and review and not review.can_leave():stay();return
        if previous==3 and label:
            label.view.set_space_down(False)
            if not label.save(only_if_dirty=True):stay();return
        page=self.pages.get(number) or self.create_page(number)
        preferred=label.mesh.path if previous==3 and label and label.mesh is not None else None
        if number==2:page.preferred_path=preferred
        self.active_page=number;self.stack.setCurrentWidget(self.surfaces[number])
        if hasattr(page,'update_shortcuts'):page.update_shortcuts()
        if hasattr(page,'shortcut_focus'):page.shortcut_focus()
        self.apply_raw(number);self.load_active()
        if number==2:page.activate(preferred)
        if number==3 and previous==2 and review and review.mesh is not None and page.root==review.root:
            if review.mesh.path in page.queue:page.list.setCurrentRow(page.queue.index(review.mesh.path))
        if number==0 and previous in (2,3) and page.root==self.root:page.refresh_surfaces()
    def review_surface_changed(self,root,path):
        label=self.pages.get(3)
        if label and label.root==Path(root):
            if label.mesh is not None and label.mesh.path==Path(path):label.reload_shared_roi()
            label.refresh_queue(changed_sources=[Path(path)])
    def load_active(self):
        if not self.root:return
        page=self.pages[self.active_page]
        loaded=getattr(page,'data_root',None) if self.active_page==1 else page.root
        if loaded==self.root:return
        if self.active_page==1:
            import orient_scanners as core
            if core.existing_stamp(self.root):
                page.status.setText('This dataset is already oriented. Use “Approved ROI dataset…” here to select the original snapshot for a new orientation review.');return
            page.load_dataset(self.root)
        else:page.load_folder(self.root)
    def choose_dataset(self):
        if self.any_busy():
            self.statusBar().showMessage('Wait for the current operation to finish.');return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'BioCal3D Approved ROI dataset',str(self.root or ''))
        if folder:
            review=self.pages.get(2)
            if review and not review.can_leave():return
            label=self.pages.get(3)
            if label and not label.save(only_if_dirty=True):return
            orientation=self.pages.get(1)
            if orientation and not orientation.confirm_discard():return
            self.remember_dataset(folder,self.active_page);self.load_active()
    def choose_raw(self):
        if self.any_busy():self.statusBar().showMessage('Wait for the current operation to finish.');return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Original full-model data folder',str(self.raw_root or ''))
        if not folder:return
        self.remember_raw(folder)
    def remember_raw(self, folder):
        review=self.pages.get(2)
        reload_review=bool(review and review.editor and review.editor.original_folder!=str(Path(folder).resolve()))
        if reload_review and not review.can_leave():return
        self.raw_root=Path(folder).resolve();key='originals/'+hashlib.sha256(str(self.root).encode()).hexdigest()[:20]
        self.settings.setValue(key,str(self.raw_root))
        for number in self.pages:self.apply_raw(number)
        if reload_review:
            review.drop_editor()
            if self.active_page==2:review.activate()
    def apply_raw(self, number):
        if not self.raw_root or number not in self.pages:return
        page=self.pages[number]
        if number==0:
            if page.original_root!=self.raw_root:
                page.original_root=self.raw_root;page.original_resolver=None;page.cache.clear();page.update_original_label()
        elif number==1:
            if not page.busy and (page.raw is None or Path(page.raw.root)!=self.raw_root):page.set_original_root(self.raw_root)
        else:
            from original_models import folder_setting_key
            if self.root:page.settings.setValue(folder_setting_key(self.root),str(self.raw_root))
    def prepare_orientation(self):
        # The copy engine rejects live annotation locks. Save and retire this
        # workspace before copying, then recreate it with fresh state on return.
        review=self.pages.get(2)
        if review and not review.can_leave():return False
        label=self.pages.get(3)
        if label:
            if not label.close():return False
            surface=self.surfaces.pop(3);self.stack.removeWidget(surface)
            self.pages.pop(3);surface.deleteLater()
        return True
    def output_ready(self, folder):
        self.oriented_output=folder;self.output_button.show();self.output_button.setToolTip(folder)
        self.statusBar().showMessage('Oriented copy ready. Click Open oriented copy to inspect or label it.')
    def use_output(self):
        self.remember_dataset(self.oriented_output,self.active_page);self.nav.setCurrentRow(0)
        self.load_active()
    def any_busy(self):
        return any(getattr(p,'busy',False) or bool(getattr(p,'tasks',{})) or getattr(p,'_loading_depth',0) for p in self.pages.values())
    def guide(self):
        QtWidgets.QMessageBox.information(self,'BioCal3D Studio — Quick guide',
            '1. Open dataset: choose Approved ROIs or an oriented copy.\n2. Original models: select the raw-data parent folder once per dataset.\n\nInspect: compare scanners and stages, change colour/geometry display, shortlist examples and export figures. Rotation, pan and zoom are linked by default.\n\nOrient: use the ORIGINAL full snapshot and TRIOS5 landmarks. Left drag rotates the view, middle drag pans, and scrolling zooms; all views are linked. Accept every scanner, save rotations, then apply to a NEW copy. Labels and approved surfaces follow the rigid rotation.\n\nReview: verify/correct ROI geometry with extraction controls and the surface brush. A/D navigate; Ctrl+Enter approves and advances. Save pending, approve, or flag the surface.\n\nLabel: use the surface annotation workflow. A/D navigate; Ctrl+S saves; Ctrl+Enter approves; hold Space to rotate. Select the correct rater before labeling.\n\nAnnotations autosave to the dataset. Inspection only reads scan data. Orientation produces a separate copy. See README.md for installation and study options.')
    def closeEvent(self,event):
        if self.any_busy():
            QtWidgets.QMessageBox.information(self,'Operation running','Let the current dataset operation finish before closing.');event.ignore();return
        orient=self.pages.get(1)
        if orient and not orient.confirm_discard():event.ignore();return
        review=self.pages.get(2)
        if review and not review.can_leave():event.ignore();return
        label=self.pages.get(3)
        if label and not label.save(only_if_dirty=True):event.ignore();return
        self.settings.setValue('geometry',self.saveGeometry())
        for page in self.pages.values():page.close()
        event.accept()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--page',choices=['inspect','orient','review','label'],default='inspect')
    parser.add_argument('--output',type=Path);parser.add_argument('--annotator');parser.add_argument('--manifest')
    parser.add_argument('--lock-annotator',action='store_true');parser.add_argument('--review-mode',action='store_true')
    parser.add_argument('--no-local-cache',action='store_true');parser.add_argument('--cache-dir',type=Path)
    args=parser.parse_args(argv)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName('BioCal3D Studio');app.setApplicationVersion('1.0.0');app.setStyle('Fusion')
    from theme import palette
    app.setPalette(palette())
    app.setWindowIcon(QtGui.QIcon(str(APP_ROOT/'assets'/'biocal3d_studio.svg')))
    window=Studio(args);window.show();return app.exec()
