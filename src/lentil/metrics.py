"""Full native-grid evaluation; unique pixels, then equal scene-group averages."""
from .common import *
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

def stitch_columns(columns):
    numerator=F.fold(columns,output_size=(1000,1080),kernel_size=(20,16),stride=(10,8))
    denominator=F.fold(torch.ones_like(columns),output_size=(1000,1080),kernel_size=(20,16),stride=(10,8))
    if not bool((denominator>0).all()):raise RuntimeError('Uncovered native pixel')
    return (numerator/denominator)[0,0]

@torch.no_grad()
def predict_frame(model,features,batch=64):
    model.eval();t=time.perf_counter()
    a=np.asarray(features,dtype=np.float32)
    if a.shape!=(16,1000,1080):raise ValueError(a.shape)
    frame=torch.from_numpy(np.array(a,copy=True))
    grid=frame.unfold(1,20,10).unfold(2,16,8).permute(1,2,0,3,4)
    h,w=grid.shape[:2];assert (h,w)==(99,134)
    logits=[];device_ms=0.;load_seconds=0.
    for start in range(0,h*w,batch):
        ix=torch.arange(start,min(start+batch,h*w));before=time.perf_counter()
        x=grid[ix//w,ix%w].contiguous().to('cuda',non_blocking=False)
        torch.cuda.synchronize();load_seconds+=time.perf_counter()-before
        ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev0.record()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        z=z.float();ev1.record();torch.cuda.synchronize();device_ms+=ev0.elapsed_time(ev1)
        if z.shape!=(len(ix),20,16) or not torch.isfinite(z).all():raise RuntimeError('Invalid model output')
        logits.append(z)
    fold_start=time.perf_counter()
    zz=torch.cat(logits).flatten(1).T.unsqueeze(0)
    out=stitch_columns(zz).cpu().numpy()
    fold_seconds=time.perf_counter()-fold_start
    return out,dict(patch_count=h*w,device_forward_ms=device_ms,batch_prepare_H2D_seconds=load_seconds,
                    stitch_D2H_seconds=fold_seconds,feature_to_CPU_logit_seconds=time.perf_counter()-t,
                    timing_scope='cached FP32 PCA feature frame to CPU logits; original source reading/PCA separately recorded')

def count_frame(r,logits,labels,compute_ap=True):
    z=np.asarray(logits,dtype=np.float32);y=np.asarray(labels);assert z.shape==y.shape==(1000,1080)
    if not np.isfinite(z).all():raise RuntimeError('Nonfinite saved logits')
    fg=y>0;pred=z>=0
    tp=int(np.count_nonzero(pred&fg));fp=int(np.count_nonzero(pred&~fg));fn=int(np.count_nonzero(~pred&fg))
    fg_n=int(fg.sum());bg_n=int((~fg).sum())
    row={k:r[k] for k in ['source','index','group','day','role','documented_normal_recording']}
    row.update(TP=tp,FP=fp,FN=fn,FG=fg_n,BG=bg_n,positive=fg_n>0,
               Dice=2*tp/(2*tp+fp+fn) if fg_n else None,IoU=tp/(tp+fp+fn) if fg_n else None,
               Recall=tp/fg_n if fg_n else None,precision=tp/(tp+fp) if tp+fp else (0. if fg_n else None),
               background_FP=fp/bg_n if bg_n else None,
               AP=float(average_precision_score(fg.ravel(),z.ravel())) if compute_ap and fg_n and bg_n else None,
               normal_status='documented_normal_recording' if r['documented_normal_recording']=='True' else ('published_zero_region_not_independently_verified' if not fg_n else 'positive_scene'))
    for category in range(1,8):
        mask=y==category;n=int(mask.sum());hit=int(np.count_nonzero(mask&pred));row[f'class{category}_pixels']=n;row[f'class{category}_TP']=hit
    return row

def aggregate(rows):
    groups={}
    for r in rows:groups.setdefault(r['group'],[]).append(r)
    summaries=[]
    for key,rr in groups.items():
        item=dict(group=key,frames=len(rr),positive=any(r['FG']>0 for r in rr))
        for metric in ['Dice','IoU','Recall','precision','AP']:
            vals=[r[metric] for r in rr if r[metric] is not None];item[metric]=float(np.mean(vals)) if vals else None
        item['FP']=sum(r['FP'] for r in rr);item['BG']=sum(r['BG'] for r in rr);item['background_FP']=item['FP']/item['BG']
        summaries.append(item)
    score={}
    positive=[r for r in summaries if r['positive']]
    for metric in ['Dice','IoU','Recall','precision','AP']:
        vals=[r[metric] for r in positive if r[metric] is not None];score[metric]=float(np.mean(vals)) if vals else None
    for label,rr in [('all',rows),('anomalous',[r for r in rows if r['FG']>0]),('zero_region',[r for r in rows if not r['FG']]),('documented_normal',[r for r in rows if r['documented_normal_recording']=='True'])]:
        bg=sum(r['BG'] for r in rr);fp=sum(r['FP'] for r in rr);score[label+'_BG']=bg;score[label+'_FP']=fp;score[label+'_background_FP']=fp/bg if bg else None
    score.update(frames=len(rows),scene_groups=len(groups),positive_groups=len(positive),TP=sum(r['TP'] for r in rows),FN=sum(r['FN'] for r in rows),FG=sum(r['FG'] for r in rows))
    for category in range(1,8):
        n=sum(r[f'class{category}_pixels'] for r in rows);hit=sum(r[f'class{category}_TP'] for r in rows)
        score[f'class{category}_pixels']=n;score[f'class{category}_TP']=hit;score[f'class{category}_Recall']=hit/n if n else None
    return score,summaries

def evaluate_dev(model,save_path=None,heartbeat=None):
    rows=members('DEV_PROPOSED');result=[];timing=[]
    logits_out=np.lib.format.open_memmap(save_path,mode='w+',dtype=np.float32,shape=(60,1000,1080)) if save_path else None
    for i,r in enumerate(rows):
        key=frame_key(r);features=np.load(ROOT/'cache/dev'/f'{key}_pca.npy',mmap_mode='r');gt=np.load(ROOT/'cache/dev'/f'{key}_labels.npy',mmap_mode='r')
        z,t=predict_frame(model,features);result.append(count_frame(r,z,gt));timing.append(dict(index=i,**t))
        if logits_out is not None:logits_out[i]=z
        if heartbeat:heartbeat(i+1,len(rows))
    if logits_out is not None:logits_out.flush();del logits_out
    score,group=aggregate(result)
    return score,result,group,timing
