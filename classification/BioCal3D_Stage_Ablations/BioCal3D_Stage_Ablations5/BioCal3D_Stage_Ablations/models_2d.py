"""Official frozen image encoders. Torch is imported only by GPU operations."""
from pathlib import Path
import hashlib, subprocess, sys
import numpy as np
from PIL import Image
from images_2d import coverage_mask,pool_numpy
from study_data import digest,signature

MODELS=['dinov2','dinov3','sam','resnet34']

def commit(path):
    try:return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return 'not-a-git-checkout'

def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        t=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()

class Encoder:
    def __init__(self,name,config,device='cuda'):
        import torch
        self.torch=torch;self.name=name;self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Use the existing CUDA-enabled biocal3d environment, or --device cpu for a slow diagnostic.')
        settings=config.get(name,{})
        self.meta={'model':name,'torch':torch.__version__,'adapter_sha256':digest(__file__),'device':str(self.device),'preprocess':'whole orthographic view; no random crop; RGB/grayscale before normalization'}
        repo=settings.get('repo','');weight=settings.get('checkpoint','')
        if name in ['dinov2','dinov3']:
            if name=='dinov3' and not (repo and weight):raise ValueError('DINOv3 requires official local repo and approved ViT-S/16 checkpoint. Set dinov3.repo and dinov3.checkpoint in model_paths.json; see README.')
            if repo:
                if not Path(repo,'hubconf.py').is_file():raise FileNotFoundError(f'Official {name} repo/hubconf.py missing: {repo}')
                kwargs={'pretrained':True}
                if name=='dinov3':kwargs['weights']=str(Path(weight).resolve())
                elif weight:kwargs['pretrained']=False
                self.model=torch.hub.load(str(Path(repo).resolve()),'dinov2_vits14' if name=='dinov2' else 'dinov3_vits16',source='local',**kwargs)
                self.meta['repo_commit']=commit(repo)
            elif name=='dinov2':
                self.model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',pretrained=not bool(weight),trust_repo=True)
                cached=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
                self.meta['repo_source']='facebookresearch/dinov2:main';self.meta['hubconf_sha256']=digest(cached/'hubconf.py')
            if name=='dinov2' and weight:
                state=torch.load(weight,map_location='cpu',weights_only=True)
                self.model.load_state_dict(state,strict=True)
            self.patch=14 if name=='dinov2' else 16
            self.meta.update(patch_size=self.patch,normalization='ImageNet mean/std',native_resize='none')
        elif name=='resnet34':
            from torchvision.models import resnet34,ResNet34_Weights
            import torchvision
            full=resnet34(weights=None if weight else ResNet34_Weights.IMAGENET1K_V1)
            if weight:full.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
            self.model=torch.nn.Sequential(*list(full.children())[:-2])
            self.meta.update(torchvision=torchvision.__version__,weights='IMAGENET1K_V1' if not weight else 'local strict state_dict',normalization='ImageNet mean/std',native_resize='none')
        elif name=='sam':
            if not weight or not Path(weight).is_file():raise FileNotFoundError('Set sam.checkpoint to official sam_vit_b_01ec64.pth in model_paths.json. See README download link.')
            if repo:sys.path.insert(0,str(Path(repo).resolve()))
            from segment_anything import sam_model_registry
            from segment_anything.utils.transforms import ResizeLongestSide
            self.sam=sam_model_registry['vit_b'](checkpoint=weight)
            self.model=self.sam.image_encoder;self.resize=ResizeLongestSide(self.model.img_size)
            self.meta.update(normalization='official SAM pixel mean/std and square padding',native_resize=int(self.model.img_size),repo_commit=commit(repo) if repo else 'installed segment_anything')
        else:raise ValueError(name)
        if weight:self.meta.update(checkpoint_sha256=digest(weight),checkpoint_file=Path(weight).name)
        self.model.eval().requires_grad_(False).to(self.device)
        if name=='sam':self.sam.eval().requires_grad_(False).to(self.device)
        self.meta['encoder_state_sha256']=state_hash(self.model)
        self.fingerprint=signature(self.meta)

    def spatial(self,rgb):
        t=self.torch
        if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:raise ValueError('Encoder input must be uint8 HWC RGB.')
        if self.name=='sam':
            resized=self.resize.apply_image(rgb)
            x=t.as_tensor(np.ascontiguousarray(resized),device=self.device).permute(2,0,1).unsqueeze(0).float()
            return self.model(self.sam.preprocess(x))
        x=t.as_tensor(np.ascontiguousarray(rgb),device=self.device).permute(2,0,1).unsqueeze(0).float()/255
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        x=(x-mean)/std
        if self.name=='resnet34':return self.model(x)
        h,w=x.shape[-2:]
        if h%self.patch or w%self.patch:raise ValueError(f'Image dimensions must be divisible by {self.patch}; use 448 or 224 (both DINO families).')
        f=self.model.forward_features(x)['x_norm_patchtokens']
        if f.shape[1]!=(h//self.patch)*(w//self.patch):raise ValueError('Unexpected patch token layout from official encoder.')
        return f.transpose(1,2).reshape(1,f.shape[-1],h//self.patch,w//self.patch)

    def extract(self,rgb,mask):
        with self.torch.no_grad():f=self.spatial(rgb)[0].float().cpu().numpy()
        if not np.isfinite(f).all():raise ValueError('Encoder produced nonfinite features.')
        coverage=coverage_mask(mask,*f.shape[1:]);embedding=pool_numpy(f,coverage)
        return embedding,f,coverage

    def close(self):
        self.model.to('cpu')
        if hasattr(self,'sam'):self.sam.to('cpu')
        if self.device.type=='cuda':self.torch.cuda.empty_cache()

def pool_torch(f,coverage):
    import torch
    # f is BCHW; coverage HW. This same operation powers classification-head CAM.
    w=torch.as_tensor(coverage,device=f.device,dtype=f.dtype)[None,None]
    mean=(f*w).sum((-2,-1))/w.sum()
    mx=f.masked_fill(w<.05,-torch.inf).amax((-2,-1))
    return torch.cat([mean,mx],dim=1)

def make_head(width,hidden=128,dropout=.25):
    import torch
    return torch.nn.Sequential(torch.nn.LayerNorm(width),torch.nn.Linear(width,hidden),torch.nn.GELU(),torch.nn.Dropout(dropout),torch.nn.Linear(hidden,3))
