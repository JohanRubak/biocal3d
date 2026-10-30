"""Find relocated scanner originals under a user-selected data root."""
from pathlib import Path,PurePosixPath
import hashlib,os,re
if __package__:
    from .performance import file_signature
else:
    from performance import file_signature

_SAMPLE=re.compile(r'BCG(\d+)T(\d+)[_-](\d+)(?!\d)',re.I)
_SCANNERS={'itero','labscanner','trios3','trios5','primescan'}

class OriginalModelLookupError(ValueError):pass

def path_parts(value):
    return tuple(p for p in PurePosixPath(str(value).replace('\\','/')).parts if p not in ('/','') and not p.endswith(':'))

def source_name(value):
    parts=path_parts(value)
    return parts[-1] if parts else ''

def identity(value):
    parts=path_parts(value);scanners=set();samples=set()
    for part in parts:
        scanner=re.sub(r'[ _-]','',part.casefold())
        if scanner in _SCANNERS:scanners.add(scanner)
        samples.update(tuple(map(int,m.groups())) for m in _SAMPLE.finditer(part))
    return scanners,samples

def folder_setting_key(root):
    token=hashlib.sha256(str(Path(root).resolve()).casefold().encode('utf-8')).hexdigest()
    return 'original_data_folders/'+token

class OriginalModelIndex:
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve();self.files=None;self.hashes={}
        if not self.root.is_dir():raise OriginalModelLookupError('Original data folder is unavailable: '+str(self.root))

    def _inside(self,path):
        return path.resolve().is_relative_to(self.root)

    def _digest(self,path):
        signature=file_signature(path)
        old=self.hashes.get(path)
        if old and old[0]==signature:return old[1]
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        value=h.hexdigest();self.hashes[path]=(signature,value)
        return value

    def _scan(self):
        files=[]
        for directory,dirs,names in os.walk(self.root,followlinks=False):
            parent=Path(directory)
            # Prepared ROI outputs are not original scanner acquisitions.
            if parent!=self.root and ('geometry_rebuild.json' in names or
                ('scan_manifest.csv' in names and (parent/'samples').is_dir())):
                dirs[:]=[];continue
            dirs[:]=[d for d in dirs if not d.startswith(('_','.')) and d.casefold() not in
                     {'annotations','annotation_output','labels','code_backups'} and not (parent/d).is_symlink()]
            for name in names:
                path=parent/name
                if path.suffix.casefold() not in ('.ply','.vtp'):continue
                low=name.casefold()
                if low in ('roi.ply','roi.vtp') or low.endswith(('.labeled.ply','.label_colors.ply','.labeled.vtp')):continue
                if self._inside(path):files.append(path)
        self.files=sorted(files,key=lambda p:str(p).casefold())

    def resolve(self,hints,*,scanner='',sample='',checksum=''):
        hints=[str(h) for h in hints if h]
        expected_scanner=re.sub(r'[ _-]','',str(scanner).casefold())
        match=_SAMPLE.search(str(sample));expected_sample=tuple(map(int,match.groups())) if match else None
        if not expected_scanner or expected_sample is None:
            for hint in hints:
                scanners,samples=identity(hint)
                if not expected_scanner and len(scanners)==1:expected_scanner=next(iter(scanners))
                if expected_sample is None and len(samples)==1:expected_sample=next(iter(samples))
        names={source_name(h).casefold() for h in hints}
        def compatible(path):
            scanners,samples=identity(path.relative_to(self.root))
            root_scanners,root_samples=identity(self.root.name)
            scanners|=root_scanners;samples|=root_samples
            return (not expected_scanner or not scanners or scanners=={expected_scanner}) and (
                expected_sample is None or not samples or samples=={expected_sample})
        # Rebase old absolute paths onto the chosen root, without a tree walk.
        direct=set()
        for hint in hints:
            parts=path_parts(hint)
            if '..' in parts:continue
            for start in range(len(parts)):
                path=self.root.joinpath(*parts[start:])
                if path.suffix.casefold() in ('.ply','.vtp') and self._inside(path) and path.is_file() and compatible(path):direct.add(path)
        def choose(paths):
            paths=sorted(set(paths),key=lambda p:str(p).casefold())
            if checksum:paths=[p for p in paths if self._digest(p).casefold()==checksum.casefold()]
            if len(paths)>1:
                raise OriginalModelLookupError('Multiple matching original models were found. Select a narrower data folder or use Locate original file…:\n'+'\n'.join(str(p) for p in paths[:8]))
            return paths[0] if paths else None
        selected=choose(direct)
        if selected:return selected
        was_cached=self.files is not None
        if self.files is None:self._scan()
        def candidates():
            result=[]
            for path in self.files:
                if not path.is_file() or not compatible(path):continue
                scanners,samples=identity(path.relative_to(self.root))
                _,root_samples=identity(self.root.name);samples|=root_samples
                # Exact sample tokens, never substring matching BCG...-1 to -10.
                if path.name.casefold() in names or (expected_sample is not None and samples=={expected_sample}):result.append(path)
            return result
        selected=choose(candidates())
        if selected:return selected
        if was_cached:
            self._scan();selected=choose(candidates())
            if selected:return selected
        detail=' with the recorded source checksum' if checksum else ''
        raise OriginalModelLookupError(f'No corresponding original model{detail} was found under {self.root}. Check the folder contains this scanner and specimen, or use Locate original file….')
