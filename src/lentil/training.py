"""Exactly eight authorised fixed-recipe models; full DEV selection, no holdout."""
from .common import *
from .networks import *
from .metrics import evaluate_dev
import gc,platform,subprocess,traceback

def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def configs():
    out=[]
    for r in read_csv(ROOT/'experiment_matrix.csv'):
        d=dict(r);d.update(seed=int(r['seed']),updates=6000,effective_batch=64,lr=float(r['lr']),run=r['model']+'_s'+r['seed'])
        d['status']='AUTHORISED';out.append(d)
    assert len(out)==8;return out
def prepare_plans():
    for seed in [42,43]:
        p=ROOT/f'plans/schedule_{seed}.npy';f=ROOT/f'plans/flips_{seed}.npy'
        if not p.exists():np.save(p,np.random.default_rng(200000+seed).integers(0,8192,size=(6000,64),dtype=np.int32))
        if not f.exists():np.save(f,np.random.default_rng(210000+seed).integers(0,4,size=(6000,64),dtype=np.uint8))
    for c in configs():
        p=ROOT/'plans'/f'initial_{c["model"]}_{c["seed"]}.pt';m=make_model(c['model'],c['seed'])
        if p.exists():assert tensor_digest(m.state_dict())==tensor_digest(torch.load(p,map_location='cpu',weights_only=True))
        else:atomic_torch(p,m.state_dict())
    atomic_json(ROOT/'locks/TRAINING_INPUT_LOCK.json',dict(created_utc=utc(),preprocessing_sha256=sha(ROOT/'preprocessing.npz'),
                schedule_sha256={str(s):sha(ROOT/f'plans/schedule_{s}.npy') for s in [42,43]},
                flips_sha256={str(s):sha(ROOT/f'plans/flips_{s}.npy') for s in [42,43]},configurations=configs()))

def real_smoke():
    if (ROOT/'SMOKE_RECEIPT.json').exists():assert read_json(ROOT/'SMOKE_RECEIPT.json')['passed'];return
    x=np.load(ROOT/'cache/train_features.npy',mmap_mode='r');y=np.load(ROOT/'cache/train_labels.npy',mmap_mode='r');records=[]
    for family in ['B1','PCA16_UNET','PCA16_T_SAME_WIDTH','B1_ALL_D1']:
        model=make_model(family,42).cuda();before=tensor_digest(model.state_dict())
        xx=torch.from_numpy(np.array(x[:4],copy=True)).cuda();yy=torch.from_numpy(np.array(y[:4],copy=True)).long().cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):zz=model(xx)
        loss,n=masked_loss(zz,yy);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert tensor_digest(model.state_dict())==before
        records.append(dict(model=family,batch=4,loss=float(loss.detach()),optimizer_updates=0,initial_unchanged=True))
        del model,xx,yy,zz,loss;gc.collect();torch.cuda.empty_cache()
    atomic_json(ROOT/'SMOKE_RECEIPT.json',dict(passed=True,records=records,optimizer_updates=0,holdout_HSI_read=0,completed_utc=utc()))

def budget_check(current_elapsed=0):
    existing=read_json(ROOT/'budget.json') if (ROOT/'budget.json').exists() else {}
    total=existing.get('completed_training_seconds',0)+current_elapsed
    if total>5*3600:raise RuntimeError('5h GPU-work budget exceeded; checkpoints preserved')

def train_one(c,xcache,ycache):
    dest=ROOT/'runs'/c['run'];dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return read_json(dest/'COMPLETE.json')
    atomic_json(dest/'config.json',c);model=make_model(c['model'],c['seed']).cuda();optimizer=optimizer_for(model,c)
    initial_path=ROOT/'plans'/f'initial_{c["model"]}_{c["seed"]}.pt';model.load_state_dict(torch.load(initial_path,map_location='cuda',weights_only=True))
    schedule=np.load(ROOT/f'plans/schedule_{c["seed"]}.npy',mmap_mode='r');flips=np.load(ROOT/f'plans/flips_{c["seed"]}.npy',mmap_mode='r')
    step=0;curve=[];per_frame=[];best=None;best_step=0;train_s=dev_s=save_s=prior_wall=0.;loss_sum=0.;loss_count=0
    if (dest/'last.pt').exists():
        state=torch.load(dest/'last.pt',map_location='cuda',weights_only=False);assert state['config']==c
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];curve=state['curve'];per_frame=state['per_frame']
        best=state['best'];best_step=state['best_step'];train_s=state['train_seconds'];dev_s=state['DEV_seconds'];save_s=state['save_seconds'];prior_wall=state['run_wall_seconds'];loss_sum=state['loss_sum'];loss_count=state['loss_count'];del state
    peak_a=max([int(r['peak_allocated_bytes']) for r in curve]+[0]);peak_r=max([int(r['peak_reserved_bytes']) for r in curve]+[0]);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    def save_resume(is_best=False,candidate_path=None):
        nonlocal save_s
        t=time.perf_counter();torch.cuda.synchronize()
        state=dict(config=c,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),curve=curve,per_frame=per_frame,best=best,best_step=best_step,
                   train_seconds=train_s,DEV_seconds=dev_s,save_seconds=save_s,run_wall_seconds=prior_wall+time.perf_counter()-start,loss_sum=loss_sum,loss_count=loss_count)
        if is_best:
            atomic_torch(dest/'best.pt',state);os.replace(candidate_path,dest/'best_DEV_logits.npy')
            write_csv(dest/'best_DEV_per_frame.csv',frames);write_csv(dest/'best_DEV_groups.csv',groups)
        atomic_torch(dest/'last.pt',state);save_s+=time.perf_counter()-t;return state
    progress('TRAINING',run=c['run'],update=step,total_updates=6000,completed_runs=len(list((ROOT/'runs').glob('*/COMPLETE.json'))))
    for update in range(step+1,6001):
        t=time.perf_counter();model.train();ix=schedule[update-1];aug=flips[update-1]
        xx=np.array(xcache[ix],copy=True);yy=np.array(ycache[ix],copy=True)
        for code in range(4):
            take=np.flatnonzero(aug==code)
            if not len(take):continue
            if code&1:xx[take]=xx[take,:,:,::-1];yy[take]=yy[take,:,::-1]
            if code&2:xx[take]=xx[take,:,::-1,:];yy[take]=yy[take,::-1,:]
        x=torch.from_numpy(xx).cuda();y=torch.from_numpy(yy).long().cuda()
        lr=learning_rate(c,update)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        loss,active=masked_loss(z,y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if update==1:
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Missing/nonfinite first gradient')
            before=next(model.parameters()).detach().clone()
        optimizer.step();torch.cuda.synchronize();step=update
        train_s+=time.perf_counter()-t;loss_sum+=float(loss.detach());loss_count+=1
        peak_a=max(peak_a,torch.cuda.max_memory_allocated());peak_r=max(peak_r,torch.cuda.max_memory_reserved())
        if update==1:
            delta=float((next(model.parameters()).detach()-before).abs().max());steps=[int(s['step'].item()) for s in optimizer.state.values()]
            assert delta>0 and min(steps)==max(steps)==1
            receipt=dict(run=c['run'],seed=c['seed'],optimizer_updates=1,patch_exposures=64,optimizer_step_min=min(steps),optimizer_step_max=max(steps),
                         first_parameter_max_abs_change=delta,loss=float(loss.detach()),created_utc=utc(),pid=os.getpid(),initial_sha256=sha(initial_path))
            atomic_json(dest/'FIRST_OPTIMIZER_UPDATE.json',receipt)
            if not (ROOT/'TRAINING_STARTED.json').exists():atomic_json(ROOT/'TRAINING_STARTED.json',receipt)
            del before
        if update%20==0 or update==1:
            budget_check(prior_wall+time.perf_counter()-start)
            progress('TRAINING',run=c['run'],update=update,total_updates=6000,loss=float(loss.detach()),lr=lr,completed_runs=len(list((ROOT/'runs').glob('*/COMPLETE.json'))),
                     peak_allocated_GiB=peak_a/2**30,peak_reserved_GiB=peak_r/2**30)
            atomic_json(dest/'progress.json',dict(update=update,loss=float(loss.detach()),train_seconds=train_s,DEV_seconds=dev_s,updated_utc=utc()))
        # The optimizer-update1000 + DEV selection is one transaction. A failed
        # DEV evaluation resumes from800 and replays the fixed prefix.
        if update%200==0 and update%1000!=0:save_resume()
        if update%1000==0:
            candidate=dest/'candidate_DEV_logits.npy';eval_start=time.perf_counter()
            def beat(i,n):progress('DEV_EVALUATION',run=c['run'],update=update,frame=i,total_frames=n,completed_runs=len(list((ROOT/'runs').glob('*/COMPLETE.json'))))
            score,frames,groups,timings=evaluate_dev(model,candidate,beat);dev_s+=time.perf_counter()-eval_start
            peak_a=max(peak_a,torch.cuda.max_memory_allocated());peak_r=max(peak_r,torch.cuda.max_memory_reserved())
            for r in frames:r.update(update=update)
            per_frame.extend(frames);d=score['Dice'];assert d is not None and np.isfinite(d)
            better=best is None or d>best
            if better:best=d;best_step=update
            curve.append(dict(update=update,lr=lr,training_loss=loss_sum/max(1,loss_count),**score,best_Dice=best,best_update=best_step,
                              train_seconds=train_s,DEV_seconds=dev_s,peak_allocated_bytes=peak_a,peak_reserved_bytes=peak_r))
            loss_sum=0.;loss_count=0;state=save_resume(better,candidate if better else None)
            if not better:candidate.unlink()
            write_csv(dest/'curve.csv',curve);write_csv(dest/'DEV_per_frame_trajectory.csv',per_frame);write_csv(dest/f'DEV_timing_update{update}.csv',timings)
            print('CHECKPOINT',c['run'],update,'DEV_DICE',d,'BEST',best_step,flush=True)
        del x,y,z,loss
    steps=[int(s['step'].item()) for s in optimizer.state.values()];assert min(steps)==max(steps)==6000 and len(curve)==6
    selected_state=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);assert selected_state['step']==best_step;del selected_state
    exposure=schedule[:best_step].ravel();counts=np.bincount(exposure,minlength=8192)
    crop_rows=read_csv(ROOT/'plans/crop_coordinates.csv');group_exposure={};source_exposure={}
    for row,n in zip(crop_rows,counts.tolist()):
        group_exposure[row['group']]=group_exposure.get(row['group'],0)+n;source_exposure[row['source']]=source_exposure.get(row['source'],0)+n
    atomic_json(dest/'BEST_EXPOSURE.json',dict(best_update=best_step,patch_exposures=int(exposure.size),by_group=group_exposure,by_source=source_exposure))
    done=dict(run=c['run'],model=c['model'],seed=c['seed'],actual_updates=6000,patch_exposures=384000,best_DEV_Dice=best,best_update=best_step,
              best_checkpoint_sha256=sha(dest/'best.pt'),last_checkpoint_sha256=sha(dest/'last.pt'),initial_sha256=sha(initial_path),
              train_seconds=train_s,DEV_seconds=dev_s,save_seconds=save_s,wall_seconds=prior_wall+time.perf_counter()-start,
              peak_allocated_bytes=peak_a,peak_reserved_bytes=peak_r,trainable_parameters=sum(p.numel() for p in model.parameters()),completed_utc=utc())
    atomic_json(dest/'COMPLETE.json',done);del model,optimizer;gc.collect();torch.cuda.empty_cache();return done

def main():
    setup_torch();lock_inputs();assert (ROOT/'PREPARATION_COMPLETE.json').exists()
    atomic_json(ROOT/'environment.json',dict(created_utc=utc(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,CUDA=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),GPU_bytes=torch.cuda.get_device_properties(0).total_memory,
                 driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),RAM_available=__import__('psutil').virtual_memory().available,
                 host_threads=4,TF32=False,BF16=True,prediction_batch=64,original_spectral_channels=61,network_feature_channels=16))
    prepare_plans();real_smoke();x=np.load(ROOT/'cache/train_features.npy',mmap_mode='r');y=np.load(ROOT/'cache/train_labels.npy',mmap_mode='r');done=[]
    for c in configs():
        complete=train_one(c,x,y);done.append(complete);write_csv(ROOT/'results/training_summary.csv',done)
        atomic_json(ROOT/'budget.json',dict(completed_training_seconds=sum(r['wall_seconds'] for r in done),completed_runs=len(done),GPU_work_limit_seconds=18000,updated_utc=utc()))
    assert len(done)==8 and sum(r['actual_updates'] for r in done)==48000
    lock=dict(created_utc=utc(),all_eight_complete=True,preprocessing_sha256=sha(ROOT/'preprocessing.npz'),selection='positive DEV group macro Dice at threshold0; earlier exact tie',
              models=[dict(run=r['run'],model=r['model'],seed=r['seed'],update=r['best_update'],checkpoint=str(ROOT/'runs'/r['run']/'best.pt'),sha256=r['best_checkpoint_sha256'],DEV_Dice=r['best_DEV_Dice']) for r in done])
    atomic_json(ROOT/'locks/MODELS_LOCK.json',lock);progress('ALL8_MODELS_LOCKED',completed_runs=8)
