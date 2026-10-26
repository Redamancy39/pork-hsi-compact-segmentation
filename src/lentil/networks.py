"""Executed lentil model configuration, reusing the public pure model bodies."""
from .common import *
import math
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from models import LightUNet,ControlStream,Block,masked_loss,poly_lr
controls=SimpleNamespace(ControlStream=ControlStream)
reference=SimpleNamespace(Block=Block,poly_lr=poly_lr)

class PCAFeatureTransformer(nn.Module):
    def __init__(self,seed):
        super().__init__()
        # Preserve the historical random-initialisation order of the common body.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(90000+seed);raw_embed=nn.Linear(184,184)
            self.blocks=nn.ModuleList([reference.Block() for _ in range(4)])
            self.final_norm=nn.LayerNorm(184,eps=1e-6);self.head=nn.Linear(184,1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(130000+seed);self.embed=nn.Linear(16,184)
        with torch.no_grad():self.embed.bias.copy_(raw_embed.bias)
    def forward(self,x):
        if x.shape[1:]!=(16,20,16):raise ValueError('Expected PCA16 native20x16 windows')
        tokens=self.embed(x.flatten(2).transpose(1,2))
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            for block in self.blocks:tokens=block(tokens)
        return self.head(self.final_norm(tokens)).squeeze(-1).reshape(x.shape[0],20,16).float()

def make_model(family,seed):
    if family=='PCA16_T_SAME_WIDTH':return PCAFeatureTransformer(seed)
    with torch.random.fork_rng(devices=[]):
        if family=='PCA16_UNET':torch.manual_seed(20000+seed);return LightUNet(16)
        if family in ['B1','B1_ALL_D1']:
            torch.manual_seed(70000+seed)
            return controls.ControlStream([1,1,2,2] if family=='B1' else [1,1,1,1],True)
    raise ValueError(family)

def learning_rate(config,update):
    base=float(config['lr'])
    if config['optimizer']=='AdamW':return reference.poly_lr(base,update)
    return base*.01+base*.99*.5*(1+math.cos(math.pi*(update-1)/5999))
def optimizer_for(model,config):
    if config['optimizer']=='AdamW':return torch.optim.AdamW(model.parameters(),lr=float(config['lr']),betas=(.9,.999),eps=1e-8,weight_decay=.0002)
    return torch.optim.Adam(model.parameters(),lr=float(config['lr']),betas=(.9,.999),eps=1e-8,weight_decay=0)
def setup_torch():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Required CUDA/BF16 unavailable')

