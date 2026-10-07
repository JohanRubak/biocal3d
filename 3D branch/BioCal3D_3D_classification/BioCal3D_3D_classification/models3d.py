"""Independent portable PyTorch implementations; architecture deviations in README."""
import torch
from torch import nn
from torch.nn import functional as F


def gather(x,idx):
    # x: B,N,C; idx: B,...; result: B,...,C
    b=torch.arange(x.shape[0],device=x.device).reshape(x.shape[0],*([1]*(idx.ndim-1)))
    return x[b,idx]

@torch.no_grad()
def fps(p,count):
    """Deterministic farthest-point sampling; pure torch, CPU or CUDA."""
    b,n,_=p.shape; count=min(count,n)
    out=torch.zeros(b,count,device=p.device,dtype=torch.long)
    distance=torch.full((b,n),float('inf'),device=p.device)
    current=(p-p.mean(1,keepdim=True)).square().sum(-1).argmax(-1)
    batch=torch.arange(b,device=p.device)
    for i in range(count):
        out[:,i]=current
        d=(p-p[batch,current][:,None]).square().sum(-1)
        distance=torch.minimum(distance,d); current=distance.argmax(-1)
    return out

@torch.no_grad()
def neighbors(q,p,k,radius=None):
    # Chunk distance matrices to bound temporary memory.
    result=[]; k=min(k,p.shape[1])
    for chunk in q.split(256,dim=1):
        d=torch.cdist(chunk,p)
        vals,idx=d.topk(k,dim=-1,largest=False,sorted=True)
        if radius is not None: idx=torch.where(vals<=radius,idx,idx[...,:1].expand_as(idx))
        result.append(idx)
    return torch.cat(result,1)

def conv1(cin,cout,act=True,bn=True):
    mods=[nn.Conv1d(cin,cout,1,bias=not bn)]
    if bn: mods.append(nn.BatchNorm1d(cout))
    if act: mods.append(nn.ReLU())
    return nn.Sequential(*mods)

def conv2(cin,cout,act=True):
    return nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU() if act else nn.Identity())

def head(cin,classes):
    # LayerNorm keeps microbatch-one training valid; deliberate change from the original BN head.
    return nn.Sequential(nn.Linear(cin,512),nn.LayerNorm(512),nn.ReLU(),nn.Dropout(.4),
                         nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(.4),nn.Linear(256,classes))

class GraphLayer(nn.Module):
    def __init__(self,cin,cout,attention):
        super().__init__(); self.attention=attention
        self.edge=conv2(cin*2,cout)
        if attention: self.score=nn.Conv2d(cin*2,cout,1)
    def forward(self,x,idx):
        near=gather(x.transpose(1,2),idx).permute(0,3,1,2)
        center=x.unsqueeze(-1).expand_as(near)
        edge=torch.cat([center-near,near],1)
        values=self.edge(edge)
        return (values*self.score(edge).softmax(-1)).sum(-1) if self.attention else values.max(-1).values

class GeometricStream(nn.Module):
    def __init__(self,attention):
        super().__init__(); self.layers=nn.ModuleList([GraphLayer(9,64,attention),GraphLayer(64,128,attention),GraphLayer(128,256,attention)])
        self.fuse=conv1(448,512)
    def forward(self,x,idx):
        features=[]
        for layer in self.layers: x=layer(x,idx); features.append(x)
        return self.fuse(torch.cat(features,1))

class TSGCClassifier(nn.Module):
    """TSGCNet-derived face classifier: geometry attention + normal max streams."""
    def __init__(self,classes=3,colour=False,k=12):
        super().__init__(); self.k=k
        self.streams=nn.ModuleList([GeometricStream(True),GeometricStream(False)]+([GeometricStream(False)] if colour else []))
        self.fusion=conv1(512*len(self.streams),512)
        self.classifier=head(1024,classes)
        # DentalPSAM-style per-mesh moments: identical normalization in train/eval.
        for module in self.modules():
            if isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):
                module.track_running_stats=False
                module.running_mean=None; module.running_var=None; module.num_batches_tracked=None
    def forward(self,batch,explain=False,ablate=None):
        p=batch['xyz']; idx=neighbors(p,p,self.k)
        local=self.fusion(torch.cat([s(x,idx) for s,x in zip(self.streams,batch['streams'])],1))
        if explain: local.retain_grad()
        used=local if ablate is None else local*(~ablate[:,None]).to(local.dtype)
        w=batch['area']; pooled=torch.cat([(used*w[:,None]).sum(-1)/w.sum(-1,keepdim=True),used.max(-1).values],1)
        logits=self.classifier(pooled)
        return (logits,local,p) if explain else logits
    def encoder_state(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('classifier.')}

class SetAbstraction(nn.Module):
    def __init__(self,cin,cout,radius,k,global_pool=False):
        super().__init__(); self.radius=radius; self.k=k; self.global_pool=global_pool
        mid=cout if global_pool else cout//2
        self.mlp=nn.Sequential(conv2(cin+3,mid),conv2(mid,cout,act=global_pool))
        if not global_pool: self.skip=conv1(cin,cout,act=False,bn=False)
    def forward(self,p,x):
        if self.global_pool:
            # OpenPoints GroupAll uses coordinates relative to the origin.
            q=p.mean(1,keepdim=True)
            dp=p.transpose(1,2).unsqueeze(2)
            grouped=x.unsqueeze(2)
            return q,self.mlp(torch.cat([dp,grouped],1)).max(-1).values
        idx=fps(p,max(1,p.shape[1]//2)); q=gather(p,idx)
        ni=neighbors(q,p,self.k,self.radius)
        dp=(gather(p,ni)-q[:,:,None])/self.radius
        grouped=gather(x.transpose(1,2),ni).permute(0,3,1,2)
        out=self.mlp(torch.cat([dp.permute(0,3,1,2),grouped],1)).max(-1).values
        skip=self.skip(gather(x.transpose(1,2),idx).transpose(1,2))
        return q,F.relu(out+skip)

class PointNeXtS(nn.Module):
    """PointNeXt-S classification topology: [1]*6 blocks, [1,2,2,2,2,1] strides."""
    def __init__(self,classes=3,colour=False,k=32,radius=.15):
        super().__init__(); self.stem=conv1(9 if colour else 6,32,act=False,bn=False)
        self.stages=nn.ModuleList([SetAbstraction(a,b,radius*1.5**i,k) for i,(a,b) in enumerate([(32,64),(64,128),(128,256),(256,512)])])
        self.global_stage=SetAbstraction(512,512,None,k,global_pool=True)
        self.classifier=head(512,classes)
    def forward(self,batch,explain=False,ablate=None):
        p=batch['xyz']; x=self.stem(batch['features'])
        for stage in self.stages: p,x=stage(p,x)
        local=x
        if explain: local.retain_grad()
        used=local if ablate is None else local*(~ablate[:,None]).to(local.dtype)
        _,global_x=self.global_stage(p,used)
        logits=self.classifier(global_x.squeeze(-1))
        return (logits,local,p) if explain else logits
    def encoder_state(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('classifier.')}

def build(model,classes,colour,k=None,radius=.15):
    return TSGCClassifier(classes,colour,k or 12) if model=='tsgcnet' else PointNeXtS(classes,colour,k or 32,radius)
