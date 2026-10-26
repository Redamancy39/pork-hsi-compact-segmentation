"""All locked models, streamed full-grid day4 evaluation with published GT."""
from .common import *
from .networks import make_model,setup_torch
from .metrics import predict_frame,count_frame,aggregate
from .preparation import iterate_frames,project
import numpy as np,torch,gc

def main():
    setup_torch();lock=read_json(ROOT/'locks/MODELS_LOCK.json');assert lock['all_eight_complete'] and len(lock['models'])==8
    assert sha(ROOT/'preprocessing.npz')==lock['preprocessing_sha256'];require_data(['DAY4_EVALUATION'])
    prep=dict(np.load(ROOT/'preprocessing.npz'));models={};entries={r['run']:r for r in lock['models']}
    for r in lock['models']:
        if sha(r['checkpoint'])!=r['sha256']:raise RuntimeError('Locked checkpoint changed')
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);assert state['step']==r['update']
        m=make_model(r['model'],r['seed']);m.load_state_dict(state['model']);m.cuda().eval();models[r['run']]=m;del state
    journal=ROOT/'results/holdout_frame_records.jsonl';records={}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);records[(r['run'],r['source'],int(r['index']))]=r
    rows=[r for r in members('DAY4_HOLDOUT_PROPOSED') if any((name,r['source'],int(r['index'])) not in records for name in models)]
    outdir=ROOT/'results/holdout_logits';outdir.mkdir(exist_ok=True);start=time.perf_counter();base_work=read_json(ROOT/'budget.json')['completed_training_seconds']
    for i,(r,a,labels,qc) in enumerate(iterate_frames(rows)) if rows else []:
        t=time.perf_counter();features=project(a,prep).transpose(2,0,1);pca_seconds=time.perf_counter()-t
        names=list(models);offset=i%len(names);names=names[offset:]+names[:offset]
        for name in names:
            key=(name,r['source'],int(r['index']))
            if key in records:continue
            torch.cuda.reset_peak_memory_stats();z,timing=predict_frame(models[name],features)
            row=count_frame(r,z,labels);row.update(run=name,model=entries[name]['model'],seed=entries[name]['seed'],**timing,
                   shared_source_read_reflectance_seconds=qc['read_reflectance_seconds'],shared_PCA_seconds=pca_seconds,
                   shared_preprocessing_note='one source/PCA read shared across locked models; not eight independently measured cu3s-to-mask workflows',
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_order=names)
            dest=outdir/f'{name}__{frame_key(r)}.npy';np.save(dest,z);row['logits_path']=str(dest);row['logits_sha256']=sha(dest)
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            records[key]=row
            progress('DAY4_EVALUATION',run=name,frame_records_completed=len(records),total_frame_records=2080,source=r['source'],index=int(r['index']),
                     elapsed_seconds=time.perf_counter()-start)
            if base_work+time.perf_counter()-start>18000:raise RuntimeError('GPU-work budget5h exceeded during holdout')
            del z
        del features;gc.collect()
    assert len(records)==2080
    summary=[]
    for name,entry in entries.items():
        rr=[r for r in records.values() if r['run']==name];assert len(rr)==260
        score,groups=aggregate(rr);summary.append(dict(run=name,model=entry['model'],seed=entry['seed'],**score,
            summed_cached_feature_to_logit_seconds=sum(r['feature_to_CPU_logit_seconds'] for r in rr),
            device_forward_seconds=sum(r['device_forward_ms'] for r in rr)/1000,peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rr)))
        write_csv(ROOT/'results'/f'{name}_holdout_per_frame.csv',rr);write_csv(ROOT/'results'/f'{name}_holdout_groups.csv',groups)
    write_csv(ROOT/'results/holdout_summary.csv',summary)
    atomic_json(ROOT/'HOLDOUT_COMPLETE.json',dict(completed_utc=utc(),frames=260,models=8,model_frame_evaluations=2080,seconds=time.perf_counter()-start,
                 normal_scope='80 published zero-region frames; no independent human certification of all normals',shared_source_preprocessing=True,
                 strict_independent_file_to_mask_benchmark=False))
    progress('HOLDOUT_COMPLETE',models=8,model_frames=2080)
