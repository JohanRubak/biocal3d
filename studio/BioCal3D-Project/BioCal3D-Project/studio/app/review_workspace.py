"""Dataset-wide ROI review workspace; plaque labels are not created here."""
from pathlib import Path
from PySide6 import QtCore,QtGui,QtWidgets
from loading_ui import LoadingFeedback,SilentLoading
from mesh_io import AnnotationStore,discover,manifest_queue
from roi_review_store import read_reviewed,read_state,resolve_review_source,current_version
from roi_review_ui import ReviewDialog
from performance import MeshCache
from local_cache import LocalReadCache
from queue_display import update_item
from theme import apply_theme


class ReviewPage(QtWidgets.QWidget):
    dataset_changed=QtCore.Signal(str)
    originals_changed=QtCore.Signal(str)
    surface_changed=QtCore.Signal(str,str)
    def __init__(self,args):
        super().__init__();apply_theme(self);self.args=args
        self.root=None;self.store=None;self.mesh=None;self.queue=[];self.index=-1;self.editor=None;self._loading_depth=0
        self.settings=QtCore.QSettings('BioCal3D','SurfaceAnnotator')
        self.mesh_cache=MeshCache();self.read_cache=None if args.no_local_cache else LocalReadCache(args.cache_dir)
        self.preferred_path=None
        layout=QtWidgets.QHBoxLayout(self);sidebar=QtWidgets.QWidget();sidebar.setFixedWidth(250);side=QtWidgets.QVBoxLayout(sidebar);layout.addWidget(sidebar)
        title=QtWidgets.QLabel('Review');title.setStyleSheet('font-size:24px;font-weight:700');side.addWidget(title)
        note=QtWidgets.QLabel('Verify and correct the shared specimen surface.');note.setWordWrap(True);side.addWidget(note)
        self.folder_label=QtWidgets.QLabel('Open an approved ROI dataset.');self.folder_label.setWordWrap(True);side.addWidget(self.folder_label)
        button=QtWidgets.QPushButton('Open dataset…');button.clicked.connect(self.choose_folder);side.addWidget(button)
        self.search=QtWidgets.QLineEdit();self.search.setPlaceholderText('Find scanner or sample…');self.search.textChanged.connect(self.filter_rows);side.addWidget(self.search)
        self.pending_only=QtWidgets.QCheckBox('Needs review only');self.pending_only.toggled.connect(self.filter_rows);side.addWidget(self.pending_only)
        self.list=QtWidgets.QListWidget();self.list.setIconSize(QtCore.QSize(18,18));self.list.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle);self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff);self.list.currentRowChanged.connect(self.select_index);side.addWidget(self.list,1)
        for text,callback in [('Previous [A]',lambda:self.navigate(-1)),('Next [D]',lambda:self.navigate(1)),('Approve & Next [Ctrl+Enter]',self.approve_next),('Next unapproved',self.next_pending)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(callback);side.addWidget(button)
            if callback==self.approve_next:button.setObjectName('primary')
        self.count=QtWidgets.QLabel('No dataset selected');self.count.setWordWrap(True);side.addWidget(self.count)
        self.content=QtWidgets.QWidget();self.content_layout=QtWidgets.QVBoxLayout(self.content);layout.addWidget(self.content,1)
        self.heading=QtWidgets.QLabel('Select a specimen to review');self.heading.setStyleSheet('font-size:20px;font-weight:600');self.heading.setWordWrap(True);self.content_layout.addWidget(self.heading)
        self.placeholder=QtWidgets.QLabel('Open a dataset to verify and correct its ROI surfaces.');self.content_layout.addWidget(self.placeholder)
        self.shortcuts=[]
        for key,callback in [('A',lambda:self.navigate(-1)),('D',lambda:self.navigate(1)),('Ctrl+Return',self.approve_next)]:
            shortcut=QtGui.QShortcut(QtGui.QKeySequence(key),self);shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut);shortcut.setAutoRepeat(False);shortcut.activated.connect(callback);self.shortcuts.append(shortcut)
        QtWidgets.QApplication.instance().focusChanged.connect(self.update_shortcuts)
    def update_shortcuts(self,*_):
        editing=isinstance(QtWidgets.QApplication.focusWidget(),(QtWidgets.QLineEdit,QtWidgets.QPlainTextEdit,QtWidgets.QTextEdit,QtWidgets.QAbstractSpinBox))
        for shortcut in self.shortcuts:shortcut.setEnabled(self.isVisible() and not editing)
        if self.editor:self.editor.update_shortcuts()
    def error(self,message):QtWidgets.QMessageBox.warning(self,'BioCal3D Studio — Review',str(message))
    def choose_folder(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Approved ROI dataset',str(self.root or ''))
        if folder:self.load_folder(Path(folder))
    def drop_editor(self):
        if self.editor:
            self.editor.dispose();self.content_layout.removeWidget(self.editor);self.editor.hide();self.editor.deleteLater();self.editor=None
    def can_leave(self):
        if self._loading_depth:return False
        if self.editor:self.editor.finish_edit()
        if not self.editor or not self.editor.has_changes():return True
        choice=QtWidgets.QMessageBox.question(self,'Unsaved ROI corrections','Save these ROI corrections as pending before continuing?',QtWidgets.QMessageBox.StandardButton.Save|QtWidgets.QMessageBox.StandardButton.Discard|QtWidgets.QMessageBox.StandardButton.Cancel)
        if choice==QtWidgets.QMessageBox.StandardButton.Cancel:return False
        if choice==QtWidgets.QMessageBox.StandardButton.Save:return bool(self.editor.commit('pending'))
        self.drop_editor();return True
    def load_folder(self,root):
        if not self.can_leave():return False
        root=Path(root).resolve()
        try:
            with LoadingFeedback(self,'Opening ROI review dataset') as feedback:
                def work():
                    store=AnnotationStore(root,self.args.output or root/'annotations','roi_review')
                    store.mesh_cache=self.mesh_cache;store.read_cache=self.read_cache
                    paths=manifest_queue(root,self.args.manifest) if self.args.manifest else discover(root,exclude_dirs=[store.output])
                    queue=list(dict.fromkeys(resolve_review_source(store,p,verify_geometry=False) for p in paths))
                    if not queue:raise ValueError('No supported ROI surfaces found in this folder.')
                    return store,queue
                store,queue=feedback.run(work,'Reading specimen inventory…')
                self.drop_editor();self.root=root;self.store=store;self.queue=queue;self.index=-1
                self.folder_label.setText(str(root));self.list.blockSignals(True);self.list.clear()
                for path in queue:self.list.addItem(str(path.relative_to(root)))
                self.list.blockSignals(False);self.dataset_changed.emit(str(root));self.refresh_rows()
                index=queue.index(self.preferred_path) if self.preferred_path in queue else next((i for i,p in enumerate(queue) if (read_state(store,p) or {}).get('status')!='approved'),0)
                self.select_index(index,confirmed=True)
            return True
        except Exception as exc:self.error(exc);return False
    def refresh_rows(self):
        if self.store is None:return
        approved=0
        for i,path in enumerate(self.queue):
            state=(read_state(self.store,path) or {}).get('status','pending');approved+=state=='approved'
            update_item(self.list.item(i),path,self.root,state,'new')
            # Geometry review has one status; labels remain a separate workspace.
            from queue_display import status_icon
            self.list.item(i).setIcon(QtGui.QIcon(status_icon(state,'new').pixmap(36,18).copy(0,0,18,18)))
            self.list.item(i).setToolTip(f'ROI: {state}\n{path}')
            self.list.item(i).setData(QtCore.Qt.ItemDataRole.AccessibleTextRole,f'ROI: {state}. {path}')
        self.count.setText(f'{approved} / {len(self.queue)} ROI surfaces approved');self.filter_rows()
    def filter_rows(self,*_):
        query=self.search.text().strip().casefold()
        for i,path in enumerate(self.queue):
            pending=(read_state(self.store,path) or {}).get('status')!='approved'
            self.list.item(i).setHidden(query not in str(path.relative_to(self.root)).casefold() or (self.pending_only.isChecked() and not pending))
    def select_index(self,index,confirmed=False):
        if index<0 or index>=len(self.queue) or self._loading_depth>1:return
        previous=self.index
        if not confirmed and not self.can_leave():
            self.list.blockSignals(True);self.list.setCurrentRow(previous);self.list.blockSignals(False);return
        try:
            with SilentLoading(self):
                mesh=read_reviewed(self.store,self.queue[index]);self.drop_editor();self.mesh=mesh;self.index=index
                self.heading.setText(f'{index+1} / {len(self.queue)} · {mesh.path.parent.name} · {mesh.path.relative_to(self.root).parent.parent}')
                self.placeholder.hide();self.editor=ReviewDialog(self,embedded=True)
                self.editor.originals_changed.connect(self.originals_changed.emit);self.editor.committed.connect(self.committed);self.editor.discard_requested.connect(self.discard_edits)
                self.content_layout.addWidget(self.editor,1);self.editor.show();self.update_shortcuts()
                self.list.blockSignals(True);self.list.setCurrentRow(index);self.list.blockSignals(False)
        except Exception as exc:self.error(exc)
    def activate(self,preferred=None):
        if preferred in self.queue and (not self.mesh or self.mesh.path!=preferred):self.select_index(self.queue.index(preferred))
        elif self.index>=0 and (self.editor is None or current_version(self.store,self.mesh.path)!=self.editor.version):self.select_index(self.index)
        self.update_shortcuts()
    def committed(self,status):
        path=self.mesh.path;self.refresh_rows()
        self.surface_changed.emit(str(self.root),str(path))
    def discard_edits(self):self.select_index(self.index,confirmed=True)
    def navigate(self,step):
        for i in range(self.index+step,len(self.queue) if step>0 else -1,step):
            if not self.list.item(i).isHidden():self.list.setCurrentRow(i);break
    def next_pending(self):
        order=list(range(self.index+1,len(self.queue)))+list(range(0,self.index+1))
        target=next((i for i in order if not self.list.item(i).isHidden() and (read_state(self.store,self.queue[i]) or {}).get('status')!='approved'),None)
        if target is not None:self.list.setCurrentRow(target)
        else:self.count.setText('All ROI surfaces in this queue are approved.')
    def approve_next(self):
        if self.editor and self.editor.commit('approved'):self.next_pending()
    def closeEvent(self,event):
        if not self.can_leave():event.ignore();return
        self.drop_editor();event.accept()
