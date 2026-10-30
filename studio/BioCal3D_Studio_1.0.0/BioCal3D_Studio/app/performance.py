"""Bounded, process-local caches; file revisions are checked on every lookup."""
from collections import OrderedDict
from copy import copy
from pathlib import Path


def file_signature(path):
    path=Path(path)
    try:
        s=path.stat()
        return (str(path),s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)
    except FileNotFoundError:
        return (str(path),None)


class MeshCache:
    """Keep at most three recently opened surfaces and 192 MiB of mesh data.

    Only immutable geometry/display inputs are cached. Labels, completion,
    locks and current review records are always loaded independently.
    """
    def __init__(self,max_bytes=192*1024*1024,max_items=3):
        self.max_bytes=max_bytes;self.max_items=max_items
        self.entries=OrderedDict();self.bytes=0

    def _size(self,mesh):
        size=sum(a.nbytes for a in [mesh.vertices,mesh.faces,mesh.active,
                 mesh.source_face_ids,mesh.source_vertex_ids] if a is not None)
        size+=int(mesh.display.GetActualMemorySize())*1024
        def extra(value):
            if isinstance(value,dict):return sum(extra(v) for v in value.values())
            if isinstance(value,(tuple,list)):return sum(extra(v) for v in value)
            if hasattr(value,'nbytes'):return value.nbytes
            if hasattr(value,'GetActualMemorySize'):return int(value.GetActualMemorySize())*1024
            return 0
        return size+extra(getattr(mesh,'_derived',{}))

    def recount(self):
        for key,(signature,mesh,_) in list(self.entries.items()):
            self.entries[key]=(signature,mesh,self._size(mesh))
        self.bytes=sum(item[2] for item in self.entries.values())
        while self.entries and (len(self.entries)>self.max_items or self.bytes>self.max_bytes):
            self.bytes-=self.entries.popitem(last=False)[1][2]

    def get(self,key,signature):
        entry=self.entries.get(key)
        if entry is None:return None
        old_signature,mesh,size=entry
        # Include the resolved texture to invalidate an image replaced in place.
        actual=signature+(file_signature(mesh.texture),) if mesh.texture else signature
        if actual!=old_signature:
            self.entries.pop(key);self.bytes-=size;return None
        self.entries.move_to_end(key)
        return copy(mesh)

    def put(self,key,signature,mesh):
        size=self._size(mesh)
        if key in self.entries:self.bytes-=self.entries.pop(key)[2]
        if size>self.max_bytes:return
        for a in (mesh.vertices,mesh.faces,mesh.active):a.flags.writeable=False
        mesh._stable_geometry=True
        # The first calculation is verified by the normal input/review checks.
        _=mesh.fingerprint
        signature=signature+(file_signature(mesh.texture),) if mesh.texture else signature
        self.entries[key]=(signature,copy(mesh),size);self.bytes+=size
        while len(self.entries)>self.max_items or self.bytes>self.max_bytes:
            self.bytes-=self.entries.popitem(last=False)[1][2]
