"""Browse paired Approved ROIs and export manuscript figure candidates."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import traceback
import roi_browser_core as core


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Approved ROIs folder')
    parser.add_argument('--output', type=Path, help='Export folder outside the dataset')
    parser.add_argument('--inventory', type=Path, help='Write scan inventory CSV and exit (no GUI)')
    parser.add_argument('--export-specimen', help='Export every available stage, e.g. BCG3-1 (no GUI)')
    parser.add_argument('--pixels', type=int, default=1600, help='Square PNG size; default 1600')
    return parser.parse_args()


def main():
    args = arguments()
    if not 400 <= args.pixels <= 6000: raise ValueError('--pixels must be between 400 and 6000')
    if args.inventory or args.export_specimen:
        if args.root is None: raise ValueError('--root is required for command-line inventory/export')
        root, scans, warnings = core.discover(args.root)
        for warning in warnings: print('Warning:', warning)
        if args.inventory:
            target = args.inventory.expanduser().resolve()
            core.safe_output(root, target.parent)
            core.write_inventory(scans, target)
            print(f'{len(scans)} scans; {len({s.physical_id for s in scans})} specimens. Inventory: {target}')
        if args.export_specimen:
            folder, metadata = core.export_set(root, args.output or Path.home() / 'Pictures' / 'BioCal3D_Figure_Examples',
                                               scans, args.export_specimen, pixels=args.pixels, progress=print)
            print(f'Exported {len(metadata["scans"])} images to {folder}')
        return
    try:
        from qt_bootstrap import prepare_qt, apply_library_paths
        qt_setup = prepare_qt()
        from PySide6 import QtCore, QtGui, QtWidgets
        apply_library_paths(QtCore, qt_setup)
        import vtk
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError as exc:
        raise RuntimeError('Activate your biocal3d environment. Missing dependency: ' + str(exc) +
                           '\nIf needed: python -m pip install -r requirements.txt') from exc

    class Signals(QtCore.QObject):
        done = QtCore.Signal(int, object)
        error = QtCore.Signal(int, str)

    class Task(QtCore.QRunnable):
        def __init__(self, token, function):
            super().__init__(); self.token = token; self.function = function; self.signals = Signals()
        def run(self):
            try: self.signals.done.emit(self.token, self.function())
            except Exception: self.signals.error.emit(self.token, traceback.format_exc())

    class View(QtWidgets.QFrame):
        def __init__(self, scanner, stage, parent):
            super().__init__(); self.scanner = scanner; self.stage = stage; self.owner = parent; self.mesh = None
            box = QtWidgets.QVBoxLayout(self); box.setContentsMargins(2, 2, 2, 2)
            header = QtWidgets.QHBoxLayout(); self.label = QtWidgets.QLabel(core.STAGE_LABELS[stage])
            self.label.setStyleSheet('font-weight:600; font-size:12px'); header.addWidget(self.label); header.addStretch()
            self.angle = QtWidgets.QSpinBox(); self.angle.setRange(-360, 360); self.angle.setSingleStep(15); self.angle.setSuffix('°')
            self.angle.setFixedWidth(65)
            self.angle.setToolTip('Display-only rotation about Z; does not register meshes or change source files.')
            self.angle.valueChanged.connect(self.redraw); header.addWidget(self.angle); box.addLayout(header)
            self.widget = QVTKRenderWindowInteractor(self); self.widget.setMinimumSize(135, 135); box.addWidget(self.widget, 1)
            self.renderer = vtk.vtkRenderer(); self.renderer.SetBackground(1, 1, 1)
            self.widget.GetRenderWindow().AddRenderer(self.renderer); self.widget.GetRenderWindow().SetMultiSamples(0)
            interactor = self.widget.GetRenderWindow().GetInteractor()
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            # Prevent VTK's W/S/R/Q keyboard bindings from changing the display silently.
            interactor.AddObserver('CharEvent', lambda obj, event: None, 1.0)
            self.widget.installEventFilter(self)
            interactor.AddObserver('InteractionEvent', lambda *_: parent.sync_views(self))
            self.widget.Initialize()
            self.info = QtWidgets.QLabel('No scan loaded'); self.info.setWordWrap(True); self.info.setMinimumHeight(48)
            self.info.setStyleSheet('font-size:11px')
            self.info.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred); box.addWidget(self.info)
        @property
        def key(self): return self.scanner + '|' + self.stage
        def eventFilter(self, obj, event):
            if event.type() in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease): return True
            return super().eventFilter(obj, event)
        def set_mesh(self, mesh, error=''):
            self.mesh = mesh; self.redraw()
            if mesh:
                self.info.setText(f'{mesh["colour_source"]}\n{mesh["vertices"]:,} vertices · {mesh["faces"]:,} triangles')
            else: self.info.setText(error or 'Missing scan for this specimen and stage')
        def redraw(self, *_):
            self.renderer.RemoveAllViewProps()
            if self.mesh:
                self.renderer.AddActor(core.actor_for(self.mesh, self.owner.mode.currentData(), self.angle.value()))
            else:
                text = vtk.vtkTextActor(); text.SetInput('No scan available')
                text.GetTextProperty().SetColor(.55, .3, .33); text.GetTextProperty().SetFontSize(18); text.SetPosition(20, 30)
                self.renderer.AddActor2D(text)
            self.renderer.ResetCameraClippingRange(); self.widget.GetRenderWindow().Render()

    class Browser(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle('Approved ROIs — Figure example browser'); self.resize(1510, 1000)
            self.settings = QtCore.QSettings('BioCal3D', 'FigureExampleBrowser')
            self.root = None; self.scans = []; self.lookup = {}; self.favorites = {}; self.cache = OrderedDict()
            self.token = 0; self.tasks = {}; self.current_id = None; self.ready = False; self.active_stages = core.STAGES
            self.pool = QtCore.QThreadPool(self); self.pool.setMaxThreadCount(1)
            central = QtWidgets.QWidget(); self.setCentralWidget(central); main = QtWidgets.QHBoxLayout(central)
            sidebar = QtWidgets.QWidget(); sidebar.setFixedWidth(260); side = QtWidgets.QVBoxLayout(sidebar); main.addWidget(sidebar)
            title = QtWidgets.QLabel('Choose a specimen'); title.setStyleSheet('font-size:19px;font-weight:600'); side.addWidget(title)
            open_button = QtWidgets.QPushButton('Open Approved ROIs folder…'); open_button.clicked.connect(self.open_folder); side.addWidget(open_button)
            self.root_label = QtWidgets.QLabel('All groups G1–G6'); self.root_label.setWordWrap(True); side.addWidget(self.root_label)
            row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel('Group'))
            self.group = QtWidgets.QComboBox(); self.group.addItem('All groups', None)
            for group in range(1, 7): self.group.addItem(f'G{group}', group)
            self.group.currentIndexChanged.connect(self.rebuild_list); row.addWidget(self.group); side.addLayout(row)
            self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText('Find specimen, e.g. BCG3-4'); self.search.textChanged.connect(self.rebuild_list); side.addWidget(self.search)
            self.only_favorites = QtWidgets.QCheckBox('Show shortlisted specimens only'); self.only_favorites.toggled.connect(self.rebuild_list); side.addWidget(self.only_favorites)
            self.list = QtWidgets.QListWidget(); self.list.currentItemChanged.connect(self.specimen_changed); side.addWidget(self.list, 1)
            nav = QtWidgets.QHBoxLayout()
            for label, step in [('← Previous [A]', -1), ('Next [D] →', 1)]:
                b = QtWidgets.QPushButton(label); b.clicked.connect(lambda _, n=step: self.navigate(n)); nav.addWidget(b)
            side.addLayout(nav)
            self.count = QtWidgets.QLabel('No dataset loaded'); self.count.setWordWrap(True); side.addWidget(self.count)
            self.notes = QtWidgets.QPlainTextEdit(); self.notes.setPlaceholderText('Why is this a useful example? Notes are saved when you click Save to shortlist or export.'); self.notes.setMaximumHeight(100); side.addWidget(self.notes)
            b = QtWidgets.QPushButton('★ Save / update shortlist'); b.clicked.connect(self.save_favorite); side.addWidget(b)
            b = QtWidgets.QPushButton('Remove from shortlist'); b.clicked.connect(self.remove_favorite); side.addWidget(b)
            content = QtWidgets.QVBoxLayout(); main.addLayout(content, 1)
            self.heading = QtWidgets.QLabel('Select your Approved ROIs dataset'); self.heading.setStyleSheet('font-size:20px;font-weight:600'); content.addWidget(self.heading)
            controls = QtWidgets.QHBoxLayout()
            self.stage = QtWidgets.QComboBox()
            for key in core.STAGES: self.stage.addItem(core.STAGE_LABELS[key], key)
            self.mode = QtWidgets.QComboBox()
            for label, value in [('Scanner colour (unshaded)', 'colour'), ('Geometry (flat shading)', 'geometry'), ('Geometry + triangle edges', 'wireframe')]: self.mode.addItem(label, value)
            self.mode.currentIndexChanged.connect(self.redraw); controls.addWidget(self.mode)
            self.texture = QtWidgets.QCheckBox('Use original texture if available'); self.texture.toggled.connect(self.load_selection); controls.addWidget(self.texture)
            controls.addStretch(); content.addLayout(controls)
            tools = QtWidgets.QHBoxLayout()
            for label, callback in [('Reset top view [R]', self.reset_top), ('Oblique view', self.oblique), ('Flip view', self.flip)]:
                b = QtWidgets.QPushButton(label); b.clicked.connect(callback); tools.addWidget(b)
            self.linked = QtWidgets.QCheckBox('Link cameras'); self.linked.setChecked(True); tools.addWidget(self.linked)
            self.fit_full = QtWidgets.QCheckBox('Fit full ROI on export'); self.fit_full.setChecked(True); tools.addWidget(self.fit_full)
            tools.addStretch(); content.addLayout(tools)
            grid = QtWidgets.QGridLayout(); content.addLayout(grid, 1)
            self.views = []; self.scanner_panels = {}
            for i, scanner in enumerate(core.SCANNERS):
                panel = QtWidgets.QGroupBox(scanner); panel.setStyleSheet('QGroupBox {font-weight:600; font-size:15px;}')
                row = QtWidgets.QHBoxLayout(panel); row.setContentsMargins(6, 20, 6, 6); row.setSpacing(5)
                self.scanner_panels[scanner] = panel
                for stage in core.STAGES:
                    view = View(scanner, stage, self); self.views.append(view); row.addWidget(view, 1)
                grid.addWidget(panel, i // 2, i % 2)
            grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
            grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
            self.notice = QtWidgets.QLabel('Left drag: rotate · Middle drag: pan · Wheel/right drag: zoom. Matching IDs do not guarantee matching in-plane orientation; check landmarks and use Rotation when needed.')
            self.notice.setWordWrap(True); content.addWidget(self.notice)
            export = QtWidgets.QHBoxLayout()
            export.addWidget(QtWidgets.QLabel('Export stage')); export.addWidget(self.stage)
            self.export_stage = QtWidgets.QPushButton('Export stage…'); self.export_stage.clicked.connect(lambda: self.export(False)); export.addWidget(self.export_stage)
            self.export_all = QtWidgets.QPushButton('Export all stages…'); self.export_all.clicked.connect(lambda: self.export(True)); export.addWidget(self.export_all)
            export.addWidget(QtWidgets.QLabel('Pixels / panel'))
            self.pixels = QtWidgets.QSpinBox(); self.pixels.setRange(400, 6000); self.pixels.setSingleStep(200); self.pixels.setValue(args.pixels); export.addWidget(self.pixels)
            b = QtWidgets.QPushButton('Export folder…'); b.clicked.connect(self.choose_output); export.addWidget(b); content.addLayout(export)
            self.output = args.output or Path(self.settings.value('output', str(Path.home() / 'Pictures' / 'BioCal3D_Figure_Examples')))
            self.output_label = QtWidgets.QLabel(str(self.output)); self.output_label.setWordWrap(True); content.addWidget(self.output_label)
            self.setStyleSheet('QWidget {font-size:13px;} QPushButton {padding:7px;} QComboBox,QLineEdit {padding:5px;} QListWidget::item {padding:8px;}')
            self.shortcuts = []
            for key, callback in [('A', lambda: self.navigate(-1)), ('D', lambda: self.navigate(1)), ('R', self.reset_top)]:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self); shortcut.activated.connect(callback); self.shortcuts.append(shortcut)
            QtWidgets.QApplication.instance().focusChanged.connect(self.shortcut_focus)
            self.set_ready(False)
            root = args.root or self.settings.value('root', '')
            if root: QtCore.QTimer.singleShot(50, lambda: self.load_folder(Path(root)))
        def shortcut_focus(self, *_):
            editing = isinstance(QtWidgets.QApplication.focusWidget(), (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox))
            for shortcut in self.shortcuts: shortcut.setEnabled(not editing)
        def set_ready(self, value):
            self.ready = value; self.export_stage.setEnabled(value); self.export_all.setEnabled(value)
        def error(self, text):
            self.statusBar().showMessage('Error'); QtWidgets.QMessageBox.warning(self, 'ROI example browser', str(text))
        def submit(self, function, callback):
            self.token += 1; token = self.token; task = Task(token, function); self.tasks[token] = task
            def done(number, result):
                self.tasks.pop(number, None)
                if number == self.token: callback(result)
            def failed(number, text):
                self.tasks.pop(number, None)
                if number == self.token: self.set_ready(False); self.error(text)
            task.signals.done.connect(done); task.signals.error.connect(failed); self.pool.start(task)
        def open_folder(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Approved ROIs', str(self.root or ''))
            if folder: self.load_folder(Path(folder))
        def load_folder(self, root):
            self.set_ready(False); self.statusBar().showMessage('Indexing sample folders…')
            self.submit(lambda: core.discover(root), self.dataset_loaded)
        def dataset_loaded(self, result):
            self.root, self.scans, warnings = result; self.lookup = {s.key: s for s in self.scans}; self.cache.clear()
            self.settings.setValue('root', str(self.root)); self.root_label.setText(str(self.root)); self.current_id = None
            self.read_favorites(); self.rebuild_list()
            self.count.setText(f'{len({s.physical_id for s in self.scans})} physical specimens\n{len(self.scans)} scans · all available G1–G6')
            if warnings: self.error('\n'.join(warnings[:20]))
        def rebuild_list(self, *_):
            if not self.scans: return
            current = self.current_id
            self.list.blockSignals(True); self.list.clear()
            specimens = sorted({(s.group, s.specimen, s.physical_id) for s in self.scans})
            for group, number, identity in specimens:
                if self.group.currentData() not in (None, group): continue
                if self.search.text().strip().lower() not in identity.lower(): continue
                if self.only_favorites.isChecked() and identity not in self.favorites: continue
                star = '★ ' if identity in self.favorites else ''
                count = sum(s.physical_id == identity for s in self.scans)
                item = QtWidgets.QListWidgetItem(f'{star}{identity}   ·   {count} scans'); item.setData(QtCore.Qt.ItemDataRole.UserRole, identity); self.list.addItem(item)
            self.list.blockSignals(False)
            selected = next((i for i in range(self.list.count()) if self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == current), 0)
            if self.list.count(): self.list.setCurrentRow(selected)
            else:
                self.token += 1; self.current_id = None; self.set_ready(False); self.heading.setText('No specimens match the filter')
                for view in self.views: view.set_mesh(None)
        def specimen_changed(self, item, old=None):
            if item is None: return
            identity = item.data(QtCore.Qt.ItemDataRole.UserRole); changed = identity != self.current_id; self.current_id = identity
            group = next(s.group for s in self.scans if s.physical_id == identity)
            self.heading.setText(f'{identity} · G{group} · {12 if group <= 2 else 24} h cultivation')
            self.stage.model().item(1).setEnabled(group > 2)
            self.active_stages = ('baseline', 'clean') if group <= 2 else core.STAGES
            for view in self.views:
                view.setVisible(view.stage in self.active_stages)
                label = {'baseline': 'Baseline', 'partial': 'Partial', 'clean': 'Final'}[view.stage]
                timepoint = {'baseline': 0, 'partial': 1, 'clean': 1 if group <= 2 else 2}[view.stage]
                view.label.setText(f'{label} (T{timepoint})')
                if view.stage not in self.active_stages: view.set_mesh(None)
            self.stage.blockSignals(True)
            if group <= 2 and self.stage.currentData() == 'partial': self.stage.setCurrentIndex(0)
            self.stage.blockSignals(False)
            if changed:
                saved = self.favorites.get(identity, {})
                self.notes.setPlainText(saved.get('note', ''))
                for view in self.views:
                    angles = saved.get('angles', {})
                    view.angle.blockSignals(True); view.angle.setValue(angles.get(view.key, angles.get(view.scanner, 0))); view.angle.blockSignals(False)
            self.load_selection()
        def navigate(self, step):
            if self.list.count(): self.list.setCurrentRow(max(0, min(self.list.count() - 1, self.list.currentRow() + step)))
        def load_selection(self, *_):
            if not self.current_id or not self.root: return
            identity, textured = self.current_id, self.texture.isChecked()
            selected = [(scanner, stage, self.lookup.get((identity, stage, scanner)))
                        for scanner in core.SCANNERS for stage in self.active_stages]
            self.set_ready(False); self.statusBar().showMessage(f'Loading {identity} / all stages…')
            for view in self.views:
                if view.stage in self.active_stages: view.set_mesh(None, 'Loading…')
            root = self.root
            def work():
                result = {}
                for scanner, stage, scan in selected:
                    pair = scanner, stage
                    if scan is None: result[pair] = (None, 'Missing scan for this specimen and stage'); continue
                    key = (str(scan.path), textured)
                    try:
                        mesh = self.cache.get(key)
                        if mesh is None: mesh = core.load_mesh(scan, root, textured); self.cache[key] = mesh
                        self.cache.move_to_end(key)
                        while len(self.cache) > 24: self.cache.popitem(last=False)
                        result[pair] = (mesh, '')
                    except Exception as exc: result[pair] = (None, str(exc))
                return result
            self.submit(work, self.selection_loaded)
        def selection_loaded(self, result):
            for view in self.views:
                mesh, error = result.get((view.scanner, view.stage), (None, 'Stage not acquired'))
                view.set_mesh(mesh, error)
            self.reset_top(); self.set_ready(any(mesh for mesh, error in result.values()))
            self.statusBar().showMessage('Ready. Scan colours are unchanged; use original textures only when available.')
        def redraw(self, *_):
            for view in self.views: view.redraw()
        def reset_top(self):
            meshes = [v.mesh for v in self.views if v.mesh]
            if not meshes: return
            camera = core.default_camera(meshes)
            for view in self.views:
                core.set_camera(view.renderer, camera); view.widget.GetRenderWindow().Render()
        def sync_views(self, source):
            if not self.linked.isChecked(): return
            camera = core.camera_state(source.renderer)
            for view in self.views:
                if view is not source: core.set_camera(view.renderer, camera); view.widget.GetRenderWindow().Render()
        def oblique(self):
            for view in self.views:
                c = view.renderer.GetActiveCamera(); c.Elevation(35); c.OrthogonalizeViewUp(); view.renderer.ResetCameraClippingRange(); view.widget.GetRenderWindow().Render()
        def flip(self):
            for view in self.views:
                c = view.renderer.GetActiveCamera(); c.Elevation(180); c.OrthogonalizeViewUp(); view.renderer.ResetCameraClippingRange(); view.widget.GetRenderWindow().Render()
        def choose_output(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose export folder outside Approved ROIs', str(self.output))
            if folder:
                try:
                    self.output = core.safe_output(self.root, folder); self.settings.setValue('output', str(self.output))
                    self.output_label.setText(str(self.output)); self.read_favorites(); self.rebuild_list()
                except Exception as exc: self.error(exc)
        def read_favorites(self):
            self.favorites = {}
            path = Path(self.output) / 'shortlist.json'
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    if data.get('dataset_root') == str(self.root): self.favorites = data.get('specimens', {})
                except Exception as exc: self.error(f'Could not read shortlist: {exc}')
        def write_favorites(self):
            output = core.safe_output(self.root, self.output)
            path = output / 'shortlist.json'; temporary = output / 'shortlist.tmp'
            temporary.write_text(json.dumps({'dataset_root': str(self.root), 'specimens': self.favorites}, indent=2), encoding='utf-8'); temporary.replace(path)
        def save_favorite(self):
            if not self.current_id: return
            try:
                self.favorites[self.current_id] = {'note': self.notes.toPlainText(), 'stage': self.stage.currentData(),
                                                  'angles': {v.key: v.angle.value() for v in self.views if v.stage in self.active_stages}}
                self.write_favorites(); self.rebuild_list(); self.statusBar().showMessage('Specimen saved to shortlist')
            except Exception as exc: self.error(exc)
        def remove_favorite(self):
            if not self.current_id: return
            try:
                self.favorites.pop(self.current_id, None); self.write_favorites(); self.rebuild_list()
            except Exception as exc: self.error(exc)
        def export(self, all_stages):
            if not self.ready: return
            if not self.linked.isChecked():
                self.error('Enable Link cameras before export. A figure set uses one consistent camera.'); return
            try:
                self.output = core.safe_output(self.root, self.output)
                dialog = QtWidgets.QProgressDialog('Preparing images…', '', 0, 0, self); dialog.setCancelButton(None); dialog.setWindowTitle('Exporting figure example')
                dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal); dialog.setMinimumDuration(0); dialog.show()
                def progress(text):
                    dialog.setLabelText(text); QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
                reference = next(v for v in self.views if v.mesh)
                camera = core.camera_state(reference.renderer)
                folder, metadata = core.export_set(self.root, self.output, self.scans, self.current_id,
                    None if all_stages else [self.stage.currentData()], camera=camera,
                    mode=self.mode.currentData(), angles={v.key: v.angle.value() for v in self.views if v.stage in self.active_stages},
                    use_texture=self.texture.isChecked(), pixels=self.pixels.value(), note=self.notes.toPlainText(), progress=progress,
                    fit_full=self.fit_full.isChecked())
                dialog.close(); self.statusBar().showMessage(f'Exported {len(metadata["scans"])} images: {folder}')
                message = f'{len(metadata["scans"])} individual PNGs, a labelled overview and export metadata saved to:\n{folder}'
                if metadata['missing_or_failed_panels']: message += f'\n\n{len(metadata["missing_or_failed_panels"])} panels were missing or failed; check the overview and metadata.'
                QtWidgets.QMessageBox.information(self, 'Export complete', message)
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
            except Exception as exc:
                if 'dialog' in locals(): dialog.close()
                self.error(traceback.format_exc())
        def closeEvent(self, event):
            self.token += 1; self.pool.clear(); self.pool.waitForDone()
            for view in self.views: view.widget.Finalize()
            event.accept()

    app = QtWidgets.QApplication(sys.argv); app.setStyle('Fusion')
    browser = Browser(); browser.show(); sys.exit(app.exec())


if __name__ == '__main__':
    try: main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
