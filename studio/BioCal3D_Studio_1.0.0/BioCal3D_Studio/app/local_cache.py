"""Bounded local read-through cache for immutable mesh/texture inputs only."""
from pathlib import Path
import hashlib,json,os,shutil,tempfile,time
if __package__:
    from .performance import file_signature
else:
    from performance import file_signature


def default_cache_root():
    base=Path((os.environ.get('LOCALAPPDATA') or Path.home()/'AppData/Local')) if os.name=='nt' else Path((os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache'))
    return base/'BioCal3D/AnnotatorCache/v1'

class LocalReadCache:
    def __init__(self,root=None,max_bytes=2*1024**3):
        self.root=Path(root or default_cache_root()).expanduser().resolve()
        self.max_bytes=max_bytes;self.verified={};self.hits=0;self.copies=0;self.last_error=''

    def _hash(self,path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()

    def materialize(self,source,progress=None):
        source=Path(source).resolve()
        if source.name=='current.json' or source.name.endswith(('.labels.npz','.lock','.labeled.ply','.label_colors.ply')):return source
        if source.suffix.lower() not in {'.ply','.vtp','.npz','.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:return source
        signature=file_signature(source)
        if signature[1] is None:raise FileNotFoundError(source)
        if source.is_relative_to(self.root) or signature[1]>self.max_bytes:return source
        token=hashlib.sha256(repr(signature).encode('utf-8')).hexdigest()
        folder=self.root/token;target=folder/source.name;metadata=folder/'entry.json'
        try:
            if metadata.is_file() and target.is_file():
                entry=json.loads(metadata.read_text(encoding='utf-8'))
                local=file_signature(target)
                if entry.get('source_signature')==list(signature) and entry.get('local_signature')==list(local):
                    if self.verified.get(token)!=local:
                        if self._hash(target)!=entry.get('sha256'):raise ValueError('Cached copy checksum differs')
                        self.verified[token]=local
                    self.hits+=1
                    try:os.utime(folder,None)
                    except OSError:pass
                    return target
        except (OSError,ValueError,KeyError):pass
        temporary=None
        try:
            folder.mkdir(parents=True,exist_ok=True)
            fd,temporary=tempfile.mkstemp(prefix='copy-',suffix='.tmp',dir=folder)
            h=hashlib.sha256();done=0
            with os.fdopen(fd,'wb') as out,source.open('rb') as inp:
                for block in iter(lambda:inp.read(1024*1024),b''):
                    out.write(block);h.update(block);done+=len(block)
                    if progress:progress('Reading '+source.name,done,signature[1])
            if file_signature(source)!=signature:
                raise RuntimeError('The source file changed while loading. Reopen the sample: '+str(source))
            os.replace(temporary,target);temporary=None
            local=file_signature(target)
            entry=dict(source_signature=list(signature),local_signature=list(local),sha256=h.hexdigest())
            fd,temp_meta=tempfile.mkstemp(prefix='metadata-',suffix='.tmp',dir=folder)
            try:
                with os.fdopen(fd,'w',encoding='utf-8') as out:json.dump(entry,out)
                os.replace(temp_meta,metadata)
            finally:
                if os.path.exists(temp_meta):os.unlink(temp_meta)
            self.verified[token]=local;self.copies+=1
            self.trim(exclude=folder)
            return target
        except OSError as error:
            # Optional acceleration must not block work when local storage is
            # full/unwritable. Source access is still required; never use stale
            # cached data when the shared drive is unavailable.
            self.last_error=str(error)
            return source
        finally:
            if temporary and os.path.exists(temporary):os.unlink(temporary)

    def trim(self,exclude=None):
        entries=[];total=0
        for folder in self.root.iterdir():
            if not folder.is_dir():continue
            try:
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                total+=size;entries.append((folder.stat().st_mtime_ns,folder,size))
            except OSError:continue
        for _,folder,size in sorted(entries):
            if total<=self.max_bytes:break
            if folder==exclude:continue
            try:
                shutil.rmtree(folder);self.verified.pop(folder.name,None);total-=size
            except OSError:pass
