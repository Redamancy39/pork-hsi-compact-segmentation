"""Two-triplet 128-file timing protocol; default invocation prints the plan only.

The measured file_run/measure operations preserve the archived measurement order.
Paths, configuration validation and environment receipts are portable wrappers.
"""
from pathlib import Path, PureWindowsPath
import argparse, csv, gc, hashlib, json, math, platform, subprocess, time

BLOCKS={
    'A':('PCA16_T_SAME_WIDTH','PCA16_DS_RESIDUAL','B1_ALL_D1'),
    'B':('B0','B1','T'),
}
BATCHES=(1,8,64)

def protocol_conditions(block):
    if block not in BLOCKS: raise ValueError('Choose execution block A or B')
    families=BLOCKS[block];rows=[]
    for batch in BATCHES:
        for repeat in range(3):
            order=families[repeat:]+families[:repeat]
            for family in order:
                roles=['FIRST_USE','WARMUP','MEASURED'] if repeat==0 else ['WARMUP','MEASURED']
                rows.extend(dict(batch=batch,repeat=repeat,family=family,role=role) for role in roles)
    return rows

def validate_config(config):
    block=config.get('execution_block')
    protocol_conditions(block)
    if config.get('seed')!=42 or config.get('batches')!=list(BATCHES):
        raise ValueError('This protocol fixes seed 42 and batches 1/8/64')
    if set(config.get('checkpoints',{}))!=set(BLOCKS[block]):
        raise ValueError('Supply exactly the three checkpoints for this block')
    if not config.get('data_roots'):raise ValueError('data_roots is required')
    for key in ['manifest','manifest_sha256','preprocessing','preprocessing_sha256']:
        if not config.get(key):raise ValueError(f'{key} is required')
    for item in config['checkpoints'].values():
        if not item.get('path') or not item.get('sha256'):raise ValueError('Checkpoint path and SHA256 required')

def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=128 or any(r.get('role')!='FIT' for r in rows):
        raise ValueError('Exactly 128 FIT members required')
    if len({r.get('image_member') for r in rows})!=128 or any(not r.get('image_member') for r in rows):
        raise ValueError('128 distinct image_member paths required')
    return rows

def resolve_member(roots,member):
    relative=Path(member)
    if relative.is_absolute() or PureWindowsPath(member).drive or '..' in relative.parts or '..' in PureWindowsPath(member).parts:
        raise ValueError('Member must be a relative path inside a data root')
    for root in roots:
        root=Path(root).resolve();candidate=(root/relative).resolve()
        if not candidate.is_relative_to(root):raise ValueError('Member leaves data root')
        if candidate.is_file():return candidate
    raise FileNotFoundError(member)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),'utf-8')
def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def checked_path(base,value,digest):
    path=(base/Path(value)).resolve()
    if not path.is_file() or sha(path)!=digest:raise ValueError(f'File identity mismatch: {path}')
    return path

def environment(np,torch):
    result=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        CUDA=torch.version.cuda,platform=platform.platform(),threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(0),GPU_bytes=torch.cuda.get_device_properties(0).total_memory,
        cudnn=torch.backends.cudnn.version(),SDPA_flags=dict(
            flash=torch.backends.cuda.flash_sdp_enabled(),memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled()),
        backend_policy='Default legal SDPA; enabled flags do not establish the selected kernel')
    for name,command in [
        ('GPU_snapshot',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used','--format=csv,noheader']),
        ('hardware_snapshot',['powershell','-NoProfile','-Command',
          '$c=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; $d=Get-CimInstance Win32_DiskDrive; @{cpu=($c|Select-Object Name,NumberOfCores,NumberOfLogicalProcessors); memory=$m.TotalPhysicalMemory; storage=($d|Select-Object Model,InterfaceType,Size)}|ConvertTo-Json -Depth 4'])]:
        try:result[name]=subprocess.check_output(command,text=True,stderr=subprocess.STDOUT,timeout=20).strip()
        except (OSError,subprocess.SubprocessError) as error:result[name]={'status':'NOT_RECORDED','reason':str(error)}
    return result

def execute(config,base,dest):
    # Heavy libraries and all device operations are confined to explicit execution.
    import numpy as np
    import torch
    from models import create_model
    validate_config(config)
    manifest=checked_path(base,config['manifest'],config['manifest_sha256'])
    prep_path=checked_path(base,config['preprocessing'],config['preprocessing_sha256'])
    specs={f:dict(s,path=checked_path(base,s['path'],s['sha256'])) for f,s in config['checkpoints'].items()}
    rows=read_manifest(manifest);roots=[(base/Path(p)).resolve() for p in config['data_roots']]
    for row in rows:resolve_member(roots,row['image_member'])
    prep=dict(np.load(prep_path,allow_pickle=False))
    dest=Path(dest).resolve();repo=Path(__file__).resolve().parents[1]
    if dest.is_relative_to(repo):raise ValueError('Use a new output directory outside the source checkout')
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'config.json',config)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Required CUDA/BF16 unavailable')
    dump(dest/'environment.json',environment(np,torch))
    dump(dest/'identities.json',dict(manifest_sha256=sha(manifest),preprocessing_sha256=sha(prep_path),
        checkpoint_sha256={f:sha(s['path']) for f,s in specs.items()},
        script_sha256=sha(Path(__file__)),models_sha256=sha(Path(__file__).with_name('models.py'))))
    paths=roots
    def host(rs):
        arrays=[]
        for r in rs:
            member=Path(r['image_member']);assert not member.is_absolute() and '..' not in member.parts
            path=next((p/member for p in paths if (p/member).is_file()),None)
            if path is None:raise FileNotFoundError(member)
            a=np.load(path,allow_pickle=False);assert a.shape==(20,16,184)
            arrays.append(np.ascontiguousarray(a.transpose(2,0,1),dtype=np.float32))
        return torch.from_numpy(np.stack(arrays))
    families=BLOCKS[config['execution_block']]
    orders=[families[i:]+families[:i] for i in range(3)]
    cohort='CURRENT_01' if config['execution_block']=='A' else 'CURRENT_02'
    raw=[];start=time.monotonic()
    try:
        for batch in BATCHES:
            for block,order in enumerate(orders):
                for family in order:
                    if time.monotonic()-start>600:raise TimeoutError('600-second execution budget reached')
                    spec=specs[family];model=create_model(prep,family,42)
                    model.load_state_dict(torch.load(spec['path'],map_location='cpu',weights_only=True)['model'],strict=True)
                    model=model.cuda().eval()
                    def file_run():
                        events=[];masks=[]
                        for off in range(0,128,batch):
                            x=host(rows[off:off+batch]).cuda()
                            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                            b.record();masks.append((z>=0).cpu().numpy());torch.cuda.synchronize();events.append(a.elapsed_time(b))
                        return math.fsum(events),np.concatenate(masks)
                    def measure(role):
                        torch.cuda.synchronize();t=time.perf_counter();event,masks=file_run()
                        elapsed=1000*(time.perf_counter()-t);assert masks.shape==(128,20,16)
                        raw.append(dict(cohort=cohort,family=family,seed=42,batch=batch,block=block,role=role,
                            patches=128,wall_ms=elapsed,device_event_ms=event,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                        write(dest/'all_passes.csv',raw)
                    if block==0:measure('FIRST_USE')
                    measure('WARMUP')
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();measure('MEASURED')
                    del model;gc.collect();torch.cuda.empty_cache()
        summary=[]
        for family in families:
            for batch in BATCHES:
                vs=[r['wall_ms'] for r in raw if r['family']==family and r['batch']==batch and r['role']=='MEASURED']
                if len(vs)!=3:raise RuntimeError('Incomplete formal condition')
                summary.append(dict(cohort=cohort,family=family,batch=batch,passes=3,
                    median_ms=float(np.median(vs)),minimum_ms=min(vs),maximum_ms=max(vs),
                    pass0_ms=vs[0],pass1_ms=vs[1],pass2_ms=vs[2]))
        write(dest/'timing_summary.csv',summary)
        dump(dest/'completion.json',dict(status='COMPLETE',file_passes=len(raw),formal_passes=27,
            execution_block=config['execution_block'],wall_seconds=time.monotonic()-start))
    except Exception as error:
        if raw:write(dest/'all_passes.csv',raw)
        dump(dest/'incomplete.json',dict(status='INCOMPLETE',error=repr(error),completed_passes=len(raw)))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text('utf-8'));validate_config(config)
    if not args.execute:
        print(json.dumps(dict(executed=False,execution_block=config['execution_block'],
            file_passes=63,formal_passes=27,scope='Two separate three-model blocks, not a six-model interleaved trial',
            schedule=protocol_conditions(config['execution_block'])),indent=2));return
    if args.output is None:parser.error('--output is required with --execute')
    execute(config,args.config.resolve().parent,args.output)

if __name__=='__main__':main()
