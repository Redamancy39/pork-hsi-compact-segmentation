"""Exact source-extracted model/metric bodies. Factory exposes the nine manuscript configurations.
Legacy helper branches are retained to avoid changing imported class definitions;
they are not exposed as release recipes. No import-time file access or execution.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
def _block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())

class LightUNet(nn.Module):
    def __init__(self,channels):
        super().__init__();self.a=_block(channels,16);self.b=_block(16,32);self.c=_block(32,64);self.upb=_block(96,32);self.upa=_block(48,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(nn.functional.max_pool2d(a,2));c=self.c(nn.functional.max_pool2d(b,2));u=self.upb(torch.cat([nn.functional.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));u=self.upa(torch.cat([nn.functional.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.out(u).squeeze(1)
class SegModel(nn.Module):
    def __init__(self,prep,variant,seed):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(prep['mean'],dtype=torch.float32)[None,:,None,None])
        self.register_buffer('scale',torch.as_tensor(prep['scale'],dtype=torch.float32)[None,:,None,None])
        w=np.array(prep['components'],dtype=np.float32,copy=True)
        if variant=='TRAINABLE_RANDOM':
            rng=np.random.default_rng(10000+seed)
            q,r=np.linalg.qr(rng.normal(size=(184,16)))
            q=q*np.where(np.diag(r)>=0,1.,-1.)[None,:]
            w=q.T.astype(np.float32)
        b=-w@np.asarray(prep['pca_mean'],dtype=np.float32)
        self.W=nn.Parameter(torch.tensor(w[:,:,None,None]),requires_grad=variant!='FIXED_PCA')
        self.b=nn.Parameter(torch.tensor(b),requires_grad=variant!='FIXED_PCA')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20000+seed);self.spatial=LightUNet(16)
        self.variant=variant
    def project(self,x):
        with torch.autocast(x.device.type,enabled=False):
            u=(x.float()-self.mean)/self.scale
            return nn.functional.conv2d(u,self.W,self.b)
    def forward(self,x):return self.spatial(self.project(x))

def masked_loss(logits,y):
    z=logits.float();valid=y>=0;target=(y==1).float()
    pos=valid&(y==1);neg=valid&(y==0)
    pc=pos.sum((1,2));nc=neg.sum((1,2));active=(pc+nc)>0
    if not active.any():raise ValueError('All-ignore batch; no valid loss/optimizer update')
    ce=nn.functional.binary_cross_entropy_with_logits(z,target,reduction='none')
    pm=(ce*pos).sum((1,2))/pc.clamp_min(1);nm=(ce*neg).sum((1,2))/nc.clamp_min(1)
    balanced=torch.where((pc>0)&(nc>0),.5*(pm+nm),torch.where(pc>0,pm,nm))
    p=z.sigmoid()*valid;gt=target*valid
    dice=1-(2*(p*gt).sum((1,2))+1)/(p.sum((1,2))+gt.sum((1,2))+1)
    return (balanced+dice)[active].mean(),int(active.sum().item())

def source_metrics(ids,logits):
    ids=np.asarray(ids).ravel();logits=np.asarray(logits,dtype=np.float32).ravel()
    valid=(ids>=0)&(ids<=12);gt=valid&(ids<10);bg=valid&(ids>=10);pred=valid&(logits>=0)
    tp=int((gt&pred).sum());fp=int((bg&pred).sum());fn=int((gt&~pred).sum());tn=int((bg&~pred).sum())
    ap=float(average_precision_score(gt[valid].astype(np.uint8),logits[valid])) if gt.any() and bg.any() else float('nan')
    d=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else float('nan');iou=tp/(tp+fp+fn) if tp+fp+fn else float('nan')
    out=dict(AP=ap,dice=d,iou=iou,precision=tp/(tp+fp) if tp+fp else float('nan'),recall=tp/(tp+fn) if tp+fn else float('nan'),TP=tp,FP=fp,FN=fn,TN=tn,foreground_pixels=tp+fn,background_pixels=fp+tn,valid_pixels=int(valid.sum()),ignore_pixels=int((~valid).sum()),background_fp_rate=fp/(fp+tn) if fp+tn else float('nan'))
    for i,name in [(10,'meat'),(11,'fat'),(12,'cb')]:
        m=ids==i;n=int(m.sum());f=int((m&pred).sum());out[name+'_bg']=n;out[name+'_fp']=f;out[name+'_fp_rate']=f/n if n else float('nan')
    for i in range(10):
        m=ids==i;n=int(m.sum());t=int((m&pred).sum());out[f'class{i}_support']=n;out[f'class{i}_TP']=t;out[f'class{i}_recall']=t/n if n else float('nan')
    return out

def make_schedule(groups,seed,updates=6000,batch=64):
    rng=np.random.default_rng(30000+seed);aug=np.random.default_rng(40000+seed)
    keys=sorted(groups);n=updates*batch;cycle=[]
    for i in range(math.ceil(n/len(keys))):cycle.extend(rng.permutation(len(keys)).tolist())
    perms={g:rng.permutation(groups[g]).tolist() for g in keys};cursor={g:0 for g in keys};out=np.empty(n,np.int32)
    for i,k in enumerate(cycle[:n]):
        g=keys[k]
        if cursor[g]==len(perms[g]):perms[g]=rng.permutation(groups[g]).tolist();cursor[g]=0
        out[i]=perms[g][cursor[g]];cursor[g]+=1
    flips=aug.integers(0,4,size=n,dtype=np.uint8)
    return out.reshape(updates,batch),flips.reshape(updates,batch)
def spectral_kernel(guide,dilation=2):
    with torch.autocast(guide.device.type,enabled=False):
        g=guide.float();n,c,h,w=g.shape
        neighbor=nn.functional.unfold(g,3,dilation=dilation,padding=dilation).reshape(n,c,9,h*w)
        distance=(neighbor-g.reshape(n,c,1,h*w)).square().sum(1)
        valid=nn.functional.unfold(torch.ones(1,1,h,w,device=g.device,dtype=torch.float32),3,dilation=dilation,padding=dilation)
        return torch.exp(-.5*distance)*valid

def pac_apply(x,weight,bias,kernel,dilation=2):
    with torch.autocast(x.device.type,enabled=False):
        n,c,h,w=x.shape;cols=nn.functional.unfold(x.float(),3,dilation=dilation,padding=dilation).reshape(n,c,9,h*w)
        inputs=(cols*kernel[:,None]).reshape(n,c*9,h*w).transpose(1,2)
        output=nn.functional.linear(inputs,weight.float().reshape(weight.shape[0],c*9),bias.float() if bias is not None else None)
        return output.transpose(1,2).reshape(n,weight.shape[0],h,w)

class PACConv(nn.Module):
    def __init__(self,conv):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias;self.dilation=conv.dilation[0];self.in_channels=conv.in_channels;self.out_channels=conv.out_channels
    def forward(self,x,kernel):return pac_apply(x,self.weight,self.bias,kernel,self.dilation)

class Residual(nn.Module):
    def __init__(self,d):
        super().__init__();self.conv1=nn.Conv2d(40,40,3,padding=d,dilation=d,bias=True);self.norm1=nn.GroupNorm(4,40)
        self.conv2=nn.Conv2d(40,40,3,padding=d,dilation=d,bias=True);self.norm2=nn.GroupNorm(4,40)
    def forward(self,x,kernel=None):
        y=self.conv1(x,kernel) if isinstance(self.conv1,PACConv) else self.conv1(x)
        y=nn.functional.relu(self.norm1(y));y=self.norm2(self.conv2(y));return nn.functional.relu(x+y)

class HighResolution(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Sequential(nn.Conv2d(16,40,1,bias=True),nn.GroupNorm(4,40),nn.ReLU())
        self.blocks=nn.ModuleList([Residual(d) for d in [1,1,2,2]]);self.out=nn.Conv2d(40,1,1,bias=True)
    def forward(self,f,kernel=None):
        x=self.stem(f)
        for block in self.blocks:x=block(x,kernel)
        return self.out(x).squeeze(1)

class HRModel(nn.Module):
    def __init__(self,prep,family,seed):
        super().__init__();assert family in ['B1','M'];self.family=family
        self.register_buffer('mean',torch.tensor(prep['mean'],dtype=torch.float32)[None,:,None,None]);self.register_buffer('scale',torch.tensor(prep['scale'],dtype=torch.float32)[None,:,None,None])
        w=np.array(prep['components'],dtype=np.float32,copy=True);self.W=nn.Parameter(torch.tensor(w[:,:,None,None]),requires_grad=False);self.b=nn.Parameter(torch.tensor(-w@np.asarray(prep['pca_mean'],dtype=np.float32)),requires_grad=False)
        with torch.random.fork_rng(devices=[]):torch.manual_seed(70000+seed);self.network=HighResolution()
        if family=='M':
            variance=np.asarray(prep['explained_variance'],dtype=np.float64);assert np.isfinite(variance).all() and (variance>=0).all()
            self.register_buffer('guide_std',torch.tensor(np.sqrt(np.maximum(variance,1e-12)),dtype=torch.float32)[None,:,None,None])
            with torch.random.fork_rng(devices=[]):torch.manual_seed(80000+seed);self.guide=nn.Conv2d(16,8,1,bias=False)
            for j in [2,3]:self.network.blocks[j].conv1=PACConv(self.network.blocks[j].conv1)
    def project(self,x):
        with torch.autocast(x.device.type,enabled=False):return nn.functional.conv2d((x.float()-self.mean)/self.scale,self.W,self.b)
    def forward(self,x):
        f=self.project(x);kernel=None
        if self.family=='M':
            with torch.autocast(x.device.type,enabled=False):g=self.guide(f/self.guide_std);kernel=spectral_kernel(g,2)
        return self.network(f,kernel).float()
def poly_lr(lr0,update):
    if not 1<=update<=6000:raise ValueError('UPDATE_OUT_OF_BUDGET')
    if update<=750:return lr0*update/750
    return lr0*.01+lr0*.99*(1-(update-750)/5250)**.9

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(184,eps=1e-6);self.qkv=nn.Linear(184,552,bias=True);self.proj=nn.Linear(184,184,bias=True);self.norm2=nn.LayerNorm(184,eps=1e-6);self.ffn=nn.Sequential(nn.Linear(184,736,bias=True),nn.GELU(),nn.Linear(736,184,bias=True))
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,8,23).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False);x=x+self.proj(a.transpose(1,2).reshape(b,n,184));return x+self.ffn(self.norm2(x))

class PixelTransformer(nn.Module):
    def __init__(self,prep,seed):
        super().__init__();self.register_buffer('mean',torch.tensor(prep['mean'],dtype=torch.float32)[None,:,None,None]);self.register_buffer('scale',torch.tensor(prep['scale'],dtype=torch.float32)[None,:,None,None])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(90000+seed);self.embed=nn.Linear(184,184,bias=True);self.blocks=nn.ModuleList([Block() for _ in range(4)]);self.final_norm=nn.LayerNorm(184,eps=1e-6);self.head=nn.Linear(184,1,bias=True)
    def forward(self,x):
        if x.shape[1:]!=(184,20,16):raise ValueError('NATIVE184x20x16_REQUIRED')
        with torch.autocast(x.device.type,enabled=False):z=(x.float()-self.mean)/self.scale
        tokens=z.flatten(2).transpose(1,2);tokens=self.embed(tokens)
        for block in self.blocks:tokens=block(tokens)
        return self.head(self.final_norm(tokens)).squeeze(-1).reshape(x.shape[0],20,16).float()

class StandardisedInput(nn.Module):
    def __init__(self,prep):
        super().__init__()
        self.register_buffer('mean',torch.tensor(prep['mean'],dtype=torch.float32)[None,:,None,None])
        self.register_buffer('scale',torch.tensor(prep['scale'],dtype=torch.float32)[None,:,None,None])
        if self.mean.shape!=(1,184,1,1) or not torch.isfinite(self.scale).all() or not (self.scale>0).all():
            raise ValueError('Frozen 184-band standardiser required')

    def standardise(self,x):
        if x.ndim!=4 or x.shape[1:]!=(184,20,16):raise ValueError('Expected N,184,20,16')
        with torch.autocast(x.device.type,enabled=False):return (x.float()-self.mean)/self.scale

class Raw184MLP(StandardisedInput):
    """Each output depends only on the 184-band spectrum at that pixel."""
    def __init__(self,prep,seed):
        super().__init__(prep)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100000+seed)
            self.network=nn.Sequential(nn.Conv2d(184,128,1),nn.ReLU(),
                nn.Conv2d(128,256,1),nn.ReLU(),nn.Conv2d(256,320,1),nn.ReLU(),nn.Conv2d(320,1,1))

    def forward(self,x):return self.network(self.standardise(x)).squeeze(1).float()

def separable(channels,dilation):
    # Linear depthwise + pointwise replaces one ordinary convolution. No new nonlinearity.
    return nn.Sequential(nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation,
                                   groups=channels,bias=False),nn.Conv2d(channels,channels,1,bias=True))

class DSBlock(nn.Module):
    def __init__(self,d):
        super().__init__();self.conv1=separable(40,d);self.norm1=nn.GroupNorm(4,40)
        self.conv2=separable(40,d);self.norm2=nn.GroupNorm(4,40)
    def forward(self,x):return F.relu(x+self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))

class DSStream(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Sequential(nn.Conv2d(16,40,1),nn.GroupNorm(4,40),nn.ReLU())
        self.blocks=nn.ModuleList([DSBlock(d) for d in (1,1,2,2)]);self.out=nn.Conv2d(40,1,1)
    def forward(self,x):
        x=self.stem(x)
        for b in self.blocks:x=b(x)
        return self.out(x).squeeze(1)

class DepthwiseResidual(StandardisedInput):
    def __init__(self,prep,seed):
        super().__init__(prep)
        w=np.asarray(prep['components'],dtype=np.float32)
        if w.shape!=(16,184):raise ValueError('Frozen PCA16 required')
        self.W=nn.Parameter(torch.tensor(w[:,:,None,None]),requires_grad=False)
        self.b=nn.Parameter(torch.tensor(-w@np.asarray(prep['pca_mean'],dtype=np.float32)),requires_grad=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(110000+seed);self.network=DSStream()
    def project(self,x):
        with torch.autocast(x.device.type,enabled=False):return F.conv2d(self.standardise(x),self.W,self.b)
    def forward(self,x):return self.network(self.project(x)).float()

def make_additional(family,prep,seed,share_b1_initial=True):
    if family=='RAW184_MLP':return Raw184MLP(prep,seed)
    if family!='PCA16_DS_RESIDUAL':raise ValueError(family)
    model=DepthwiseResidual(prep,seed)
    if share_b1_initial:
        initial=HRModel(prep, 'B1', seed).state_dict()
        own=model.state_dict()
        for key in own:
            if key.startswith('network.') and ('.stem.' in key or '.norm' in key or '.out.' in key):
                if key not in initial or own[key].shape!=initial[key].shape:raise ValueError('Shared B1 initialization mismatch: '+key)
                own[key]=initial[key].clone()
        model.load_state_dict(own)
    return model

class RawUNet(StandardisedInput):
    def __init__(self,prep,seed):
        super().__init__(prep)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(120000+seed);self.spatial=LightUNet(184)
        old=SegModel(prep, 'FIXED_PCA', seed).state_dict()
        state=self.state_dict()
        for k in state:
            if k=='spatial.a.0.weight':continue
            if k not in old or old[k].shape!=state[k].shape:raise ValueError('Shared B0 initial mismatch: '+k)
            state[k]=old[k].clone()
        self.load_state_dict(state)
    def forward(self,x):return self.spatial(self.standardise(x)).float()

class ControlBlock(nn.Module):
    def __init__(self,d,use_skip):
        super().__init__();self.use_skip=use_skip
        self.conv1=nn.Conv2d(40,40,3,padding=d,dilation=d,bias=True);self.norm1=nn.GroupNorm(4,40)
        self.conv2=nn.Conv2d(40,40,3,padding=d,dilation=d,bias=True);self.norm2=nn.GroupNorm(4,40)
    def forward(self,x):
        y=self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x)))))
        return F.relu(x+y if self.use_skip else y)

class ControlStream(nn.Module):
    def __init__(self,dilations,use_skip):
        super().__init__();self.stem=nn.Sequential(nn.Conv2d(16,40,1),nn.GroupNorm(4,40),nn.ReLU())
        self.blocks=nn.ModuleList([ControlBlock(d,use_skip) for d in dilations]);self.out=nn.Conv2d(40,1,1)
    def forward(self,x):
        x=self.stem(x)
        for block in self.blocks:x=block(x)
        return self.out(x).squeeze(1)

class ControlledResidual(StandardisedInput):
    def __init__(self,prep,seed,dilations,use_skip):
        super().__init__(prep)
        w=np.array(prep['components'],dtype=np.float32,copy=True)
        if w.shape!=(16,184):raise ValueError('Frozen PCA16 required')
        self.W=nn.Parameter(torch.tensor(w[:,:,None,None]),requires_grad=False)
        self.b=nn.Parameter(torch.tensor(-w@np.asarray(prep['pca_mean'],dtype=np.float32)),requires_grad=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(70000+seed);self.network=ControlStream(dilations,use_skip)
    def project(self,x):
        with torch.autocast(x.device.type,enabled=False):return F.conv2d(self.standardise(x),self.W,self.b)
    def forward(self,x):return self.network(self.project(x)).float()

def make_control(family,prep,seed):
    if family=='RAW184_UNET':return RawUNet(prep,seed)
    if family=='B1_ALL_D1':m=ControlledResidual(prep,seed,[1,1,1,1],True)
    elif family=='B1_NO_SKIP':m=ControlledResidual(prep,seed,[1,1,2,2],False)
    else:raise ValueError(family)
    m.load_state_dict(HRModel(prep, 'B1', seed).state_dict())
    return m

class PCA16Transformer(PixelTransformer):
    def __init__(self,prep,seed):
        super().__init__(prep,seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(130000+seed);self.embed=nn.Linear(16,184,bias=True)
        original=PixelTransformer(prep, seed).state_dict()
        frozen=HRModel(prep, 'B1', seed).state_dict()
        for k in ['mean','scale']:
            if not torch.equal(original[k],self.state_dict()[k]) or not torch.equal(original[k],frozen[k]):
                raise ValueError('Frozen FIT standardiser identity mismatch')
        expected=torch.tensor(np.asarray(prep['components'],dtype=np.float32)[:,:,None,None])
        if not torch.equal(frozen['W'],expected):raise ValueError('Frozen PCA matrix identity mismatch')
        self.W=nn.Parameter(frozen['W'].clone(),requires_grad=False)
        self.b=nn.Parameter(frozen['b'].clone(),requires_grad=False)
        state=self.state_dict()
        for k,v in original.items():
            if k=='embed.weight':continue
            if k not in state or state[k].shape!=v.shape:raise ValueError('Shared T initial mismatch: '+k)
            state[k]=v.clone()
        self.load_state_dict(state)
    def project(self,x):
        if x.shape[1:]!=(184,20,16):raise ValueError('NATIVE184x20x16_REQUIRED')
        with torch.autocast(x.device.type,enabled=False):
            return F.conv2d((x.float()-self.mean)/self.scale,self.W,self.b)
    def forward(self,x):
        tokens=self.project(x).flatten(2).transpose(1,2);tokens=self.embed(tokens)
        for block in self.blocks:tokens=block(tokens)
        return self.head(self.final_norm(tokens)).squeeze(-1).reshape(x.shape[0],20,16).float()

FAMILIES = ['B0', 'B1', 'T', 'RAW184_MLP', 'PCA16_DS_RESIDUAL', 'RAW184_UNET', 'B1_ALL_D1', 'B1_NO_SKIP', 'PCA16_T_SAME_WIDTH']

def create_model(prep, family, seed):
    if family == 'B0': return SegModel(prep, 'FIXED_PCA', seed)
    if family == 'B1': return HRModel(prep, 'B1', seed)
    if family == 'T': return PixelTransformer(prep, seed)
    if family in ['RAW184_MLP','PCA16_DS_RESIDUAL']:
        return make_additional(family, prep, seed)
    if family in ['RAW184_UNET','B1_ALL_D1','B1_NO_SKIP']:
        return make_control(family, prep, seed)
    if family == 'PCA16_T_SAME_WIDTH': return PCA16Transformer(prep, seed)
    raise ValueError(family)


def load_preprocessing(assets_root):
    """Load this release's frozen FIT statistics and archived projection biases."""
    from pathlib import Path
    import json
    root=Path(assets_root)
    prep=dict(np.load(root/'artifacts/preprocessing.npz',allow_pickle=False))
    bias_path=root/'artifacts/projection_biases.json'
    if bias_path.is_file():prep['archived_projection_biases']=json.loads(bias_path.read_text())
    return prep

_create_model_body=create_model

def create_model(prep,family,seed):
    model=_create_model_body(prep,family,seed)
    bias=prep.get('archived_projection_biases',{}).get(f'{family}_{seed}')
    if bias is not None:
        with torch.no_grad():model.b.copy_(torch.tensor(bias,dtype=torch.float32))
    return model
