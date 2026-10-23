"""Portable I/O glue around unchanged research model/metric bodies.

Portable entry points have lightweight checks, not a full training replay.
No work occurs on import. Historical data and experiment assets are supplied externally.
"""
from pathlib import Path
import argparse, csv, json, math, time, collections, gc
import numpy as np
import torch
from PIL import Image
from models import create_model, masked_loss, source_metrics, poly_lr, FAMILIES, load_preprocessing
from labels import decode_label
ROOT=Path(__file__).resolve().parents[1]
ASSETS=None

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def member_path(data_root,member):
    root=Path(data_root).resolve();p=(root/member).resolve()
    if not p.is_relative_to(root):raise ValueError('Member leaves data root')
    if not p.is_file():raise FileNotFoundError(p)
    return p

class Batches:
    def __init__(self,data_root,rows):
        self.data_root=Path(data_root);self.rows=rows;self.cache=collections.OrderedDict()
        self.mapping=json.loads((ASSETS/'manifests/label_mapping.json').read_text('utf-8'))
    def pair(self,i):
        i=int(i)
        if i not in self.cache:
            r=self.rows[i];x=np.load(member_path(self.data_root,r['image_member']),allow_pickle=False)
            p=member_path(self.data_root,r['label_member'])
            if p.suffix=='.npy':a=np.load(p,allow_pickle=False)
            else:
                with Image.open(p) as im:a=np.asarray(im).copy()
            ids,y,_=decode_label(a,self.mapping)
            assert x.shape==(20,16,184) and ids.shape==(20,16)
            self.cache[i]=(np.ascontiguousarray(x.transpose(2,0,1),dtype=np.float32),ids,y)
            if len(self.cache)>1024:self.cache.popitem(last=False)
        else:self.cache.move_to_end(i)
        return self.cache[i]
    def get(self,indices,flips=None,fit=False):
        xx=[];yy=[]
        for j,i in enumerate(indices):
            if fit:assert self.rows[int(i)].get('role')=='FIT'
            x,_,y=self.pair(i);f=int(flips[j]) if flips is not None else 0
            if f&1:x=x[:,::-1,:];y=y[::-1,:]
            if f&2:x=x[:,:,::-1];y=y[:,::-1]
            xx.append(x);yy.append(y)
        return torch.from_numpy(np.stack(xx)).cuda(),torch.from_numpy(np.stack(yy)).cuda()

def setup():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def load_model(family,seed,checkpoint=None):
    prep=load_preprocessing(ASSETS)
    model=create_model(prep,family,seed)
    if checkpoint is not None:
        state=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        model.load_state_dict(state.get('model',state), strict=True)
    return model.cuda()

def evaluate(model,batches,indices):
    model.eval();pred=np.empty((len(indices),20,16),np.float32)
    with torch.inference_mode():
        for off in range(0,len(indices),64):
            selected=indices[off:off+64];x,_=batches.get(selected)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
            pred[off:off+len(selected)]=z.float().cpu().numpy()
    out=[]
    for parent in dict.fromkeys(batches.rows[i]['parent'] for i in indices):
        pos=[j for j,i in enumerate(indices) if batches.rows[i]['parent']==parent]
        ids=np.stack([batches.pair(indices[j])[1] for j in pos])
        out.append(dict(parent=parent,**source_metrics(ids,pred[pos])))
    return out,pred

def train(data_root,output,family,seed,lr0):
    run=f'{family}_lr{lr0}_s{seed}'
    config=dict(run=run,family=family,seed=seed,initial_lr=lr0,updates=6000,effective_batch=64)
    rows=read_csv(ASSETS/'manifests/fit_dev.csv');assert len(rows)==8192
    batches=Batches(data_root,rows);dev=[i for i,r in enumerate(rows) if r['role']=='DEV']
    plan=np.load(ASSETS/f'artifacts/plans/schedule_{seed}.npy');flips=np.load(ASSETS/f'artifacts/plans/flips_{seed}.npy')
    assert plan.shape==flips.shape==(6000,64) and all(rows[int(i)]['role']=='FIT' for i in np.unique(plan))
    model=load_model(family,seed)
    parameters=[p for p in model.parameters() if p.requires_grad]
    cls=torch.optim.AdamW if family in ['T','PCA16_T_SAME_WIDTH'] else torch.optim.Adam
    opt=cls(parameters,lr=lr0,betas=(.9,.999),eps=1e-8,weight_decay=.0002 if family in ['T','PCA16_T_SAME_WIDTH'] else 0.)
    curve=[];source_curve=[];best=-float('inf');apbest=-float('inf');loss_total=0.
    for update in range(1,6001):
        model.train();lr=poly_lr(lr0,update) if family in ['T','PCA16_T_SAME_WIDTH'] else lr0*.01+lr0*.99*.5*(1+math.cos(math.pi*(update-1)/5999))
        for g in opt.param_groups:g['lr']=lr
        x,y=batches.get(plan[update-1],flips[update-1],fit=True);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        loss,n=masked_loss(z,y);assert torch.isfinite(loss) and n==64
        loss.backward();opt.step();loss_total+=float(loss.detach())
        if update%200==0:
            sr,pred=evaluate(model,batches,dev)
            macro={k:float(np.mean([r[k] for r in sr])) for k in ['AP','dice','iou','precision','recall','meat_fp_rate','fat_fp_rate','cb_fp_rate']}
            curve.append(dict(update=update,lr=lr,train_loss=loss_total/200,**macro));loss_total=0.
            source_curve.extend(dict(update=update,**r) for r in sr)
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=update,config=config)
            torch.save(state,output/'last.pt')
            for name,key,current in [('dice_best','dice',best),('ap_best','AP',apbest)]:
                if macro[key]>current:
                    torch.save(state,output/(name+'.pt'));np.save(output/(name+'_logits.npy'),pred)
                    write_csv(output/(name+'_per_source.csv'),sr)
                    if key=='dice':best=macro[key]
                    else:apbest=macro[key]
            write_csv(output/'curve.csv',curve);write_csv(output/'source_curve.csv',source_curve)
            print(run,update,macro,flush=True)

def score_comparative(data_root,output,family,seed,checkpoint):
    # Historical file role is preserved in manifest; current use is comparative.
    rows=read_csv(ASSETS/'manifests/comparative.csv');assert len(rows)==2048
    batches=Batches(data_root,rows);m=load_model(family,seed,checkpoint)
    scores,z=evaluate(m,batches,list(range(len(rows))))
    write_csv(output/'per_source.csv',scores);np.save(output/'logits.npy',z)

def timing(data_root,output,checkpoint_dir):
    rows=read_csv(ASSETS/'manifests/timing128.csv');assert len(rows)==128 and all(r['role']=='FIT' for r in rows)
    def host(rs):return torch.from_numpy(np.stack([np.ascontiguousarray(np.load(member_path(data_root,r['image_member']),allow_pickle=False).transpose(2,0,1),dtype=np.float32) for r in rs]))
    raw=[];first=[];orders=[['B0','B1','T'],['B1','T','B0'],['T','B0','B1']]
    for mode in ['DEVICE_FORWARD','NPY_FILE_TO_CPU_MASK']:
        for batch in [1,8,64]:
            x=host(rows[:batch]).cuda() if mode=='DEVICE_FORWARD' else None
            for block,order in enumerate(orders):
                for family in order:
                    model=load_model(family,42,checkpoint_dir/f'{family}_42.pt').eval()
                    def invoke():
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                        b.record();torch.cuda.synchronize();return a.elapsed_time(b)
                    def file_run():
                        masks=[];events=[]
                        for off in range(0,128,batch):
                            xx=host(rows[off:off+batch]).cuda();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(xx)
                            b.record();masks.append((z>=0).cpu().numpy());torch.cuda.synchronize();events.append(a.elapsed_time(b))
                        assert np.concatenate(masks).shape==(128,20,16)
                        return math.fsum(events)
                    fn=invoke if mode=='DEVICE_FORWARD' else file_run
                    if block==0:
                        t=time.perf_counter();ev=fn();first.append(dict(mode=mode,family=family,batch=batch,wall_ms=1000*(time.perf_counter()-t),device_event_ms=ev))
                    for _ in range(20 if mode=='DEVICE_FORWARD' else 1):fn()
                    for j in range(100 if mode=='DEVICE_FORWARD' else 1):
                        t=time.perf_counter();ev=fn();raw.append(dict(mode=mode,family=family,batch=batch,block=block,iteration=j,patches=batch if mode=='DEVICE_FORWARD' else 128,wall_ms=1000*(time.perf_counter()-t),device_event_ms=ev))
                    del model;gc.collect();torch.cuda.empty_cache()
            del x
    write_csv(output/'timing_raw.csv',raw);write_csv(output/'timing_first_call.csv',first)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    global ASSETS
    p.add_argument('action',choices=['train','evaluate','time']);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--assets-root',type=Path,required=True,help='External preprocessing, manifests and plans; not distributed in this code-only repository')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--learning-rate',type=float);p.add_argument('--family',choices=FAMILIES);p.add_argument('--seed',type=int,choices=[42,43,44,45]);p.add_argument('--execute',action='store_true')
    p.add_argument('--checkpoint', type=Path);p.add_argument('--checkpoint-dir', type=Path)
    a=p.parse_args();ASSETS=a.assets_root.resolve()
    if not a.execute:p.error('Explicit --execute required. Full execution has not been validated for this candidate.')
    if a.action=='train' and (not a.family or a.seed is None or a.learning_rate is None or a.learning_rate<=0):p.error('--family, --seed and positive --learning-rate are required')
    if a.action=='evaluate' and (not a.family or not a.seed):p.error('--family and --seed required')
    if a.output.resolve().is_relative_to(ROOT.resolve()):p.error('Use a new output directory outside the release candidate')
    if a.action=='evaluate' and not a.checkpoint:p.error('--checkpoint is required; trained weights are not bundled')
    if a.action=='time' and not a.checkpoint_dir:p.error('--checkpoint-dir is required')
    a.output.mkdir(parents=True,exist_ok=False);setup()
    if a.action=='train':train(a.data_root,a.output,a.family,a.seed,a.learning_rate)
    elif a.action=='evaluate':score_comparative(a.data_root,a.output,a.family,a.seed,a.checkpoint)
    else:timing(a.data_root,a.output,a.checkpoint_dir)
if __name__=='__main__':main()
