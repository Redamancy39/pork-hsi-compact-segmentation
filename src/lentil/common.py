"""Portable paths, protocol inputs and durable records for the lentil workflow."""
from pathlib import Path
import json,csv,hashlib,os,time,datetime,shutil

ROOT=None
METADATA=None
DATA_ROOT=None
SETTINGS={}
REVISION='ce3a213e96f83b0b2499514c09a661e52332110e'
ROLES={'FIT':'FIT_PROPOSED','DEV':'DEV_PROPOSED','DAY4_EVALUATION':'DAY4_HOLDOUT_PROPOSED',
       'FIT_PROPOSED':'FIT_PROPOSED','DEV_PROPOSED':'DEV_PROPOSED','DAY4_HOLDOUT_PROPOSED':'DAY4_HOLDOUT_PROPOSED'}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r));tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seeded_int(value):return int(hashlib.sha256(value.encode()).hexdigest()[:16],16)
def canonical_role(role):
    if role not in ROLES:raise ValueError('Unknown member role: '+str(role))
    return ROLES[role]
def contained_path(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if not p.is_relative_to(root):raise ValueError('Member leaves input root')
    return p
def validate_members(rows,require_counts=True):
    groups={};keys=set();counts={r:0 for r in set(ROLES.values())}
    for row in rows:
        role=canonical_role(row['role']);key=(row['source'],int(row['index']))
        if key in keys:raise ValueError('Duplicate source/frame member')
        keys.add(key);counts[role]+=1
        source=Path(row['source'])
        if source.is_absolute() or '..' in source.parts:raise ValueError('Unsafe relative source path')
        groups.setdefault(row['group'],set()).add(role)
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Scene group shared across research roles')
    if require_counts:
        if counts!={'FIT_PROPOSED':340,'DEV_PROPOSED':60,'DAY4_HOLDOUT_PROPOSED':260}:raise ValueError('Fixed660-member protocol required')
        for group in groups:
            if sum(r['group']==group for r in rows)!=4:raise ValueError('Four lighting views required per group')
        for row in rows:
            role=canonical_role(row['role'])
            if (row['day']=='day4')!=(role=='DAY4_HOLDOUT_PROPOSED'):raise ValueError('Held-out date assignment changed')
    return rows
def recipe_rows():
    out=[]
    for family in ['B1','PCA16_UNET','PCA16_T_SAME_WIDTH','B1_ALL_D1']:
        for seed in [42,43]:
            transformer=family=='PCA16_T_SAME_WIDTH'
            out.append(dict(model=family,seed=seed,lr=.0003 if family=='PCA16_UNET' else .001,
                optimizer='AdamW' if transformer else 'Adam',schedule='warmup750_poly0.9' if transformer else 'COSINE',
                trainable_parameters=1638337 if transformer else (120785 if family=='PCA16_UNET' else 116961),
                updates=6000,effective_batch=64,patch_height=20,patch_width=16,DEV_every_updates=1000,DEV_candidates=6,
                selection='DEV positive-scene-group macro Dice at logit>=0; exact tie earlier',status='FIXED_REPRODUCTION_RECIPE'))
    return out
def configure(config,execute=False):
    global ROOT,METADATA,DATA_ROOT,SETTINGS
    path=Path(config).resolve();s=read_json(path)
    if s.get('dataset_revision')!=REVISION:raise ValueError('Pinned dataset revision required')
    for k in ['data_root','metadata_root','members_csv','work_root']:
        if not isinstance(s.get(k),str) or not s[k]:raise ValueError('Missing path: '+k)
        s[k]=str((path.parent/s[k]).resolve())
    if s.get('cuvis_bin'):s['cuvis_bin']=str((path.parent/s['cuvis_bin']).resolve())
    ROOT=Path(s['work_root']);METADATA=Path(s['metadata_root']);DATA_ROOT=Path(s['data_root']);SETTINGS.clear();SETTINGS.update(s)
    if execute:
        checkout=Path(__file__).resolve().parents[2]
        if ROOT.is_relative_to(checkout):raise ValueError('Work directory must be outside source checkout')
        for name in ['logs','cache','plans','runs','results','qc','locks']:(ROOT/name).mkdir(parents=True,exist_ok=True)
        lock_inputs()
    return s
def members(role=None):
    rr=read_csv(ROOT/'members.csv')
    return [r for r in rr if role is None or r['role']==canonical_role(role)]
def frame_key(r):return Path(r['source']).stem+'__'+str(int(r['index'])).zfill(4)
def container_path(source):return contained_path(DATA_ROOT,source)
def require_data(roles):
    for role in roles:
        for row in members(role):
            for p in [container_path(row['source']),contained_path(METADATA,Path(row['source']).with_suffix('.json'))]:
                if not p.is_file():raise FileNotFoundError(p)
def progress(stage,**fields):
    data=dict(stage=stage,updated_utc=utc(),**fields);atomic_json(ROOT/'progress.json',data)
    print(utc(),stage,json.dumps(fields,ensure_ascii=False),flush=True)
def lock_inputs():
    source=Path(SETTINGS['members_csv']);rr=validate_members(read_csv(source))
    for r in rr:r['role']=canonical_role(r['role'])
    dest=ROOT/'members.csv'
    if dest.exists():
        if read_csv(dest)!=rr:raise ValueError('Existing work directory has different members')
    else:write_csv(dest,rr)
    matrix=ROOT/'experiment_matrix.csv'
    if not matrix.exists():write_csv(matrix,recipe_rows())
    else:
        expected=[{k:str(v) for k,v in r.items()} for r in recipe_rows()]
        if read_csv(matrix)!=expected:raise ValueError('Existing work directory has different recipes')
    inputs=dict(revision=REVISION,published_members_sha256=sha(source),member_sha256=sha(dest),matrix_sha256=sha(matrix))
    p=ROOT/'locks/INPUT_LOCK.json'
    if p.exists() and read_json(p)!=inputs:raise ValueError('Input identity changed')
    atomic_json(p,inputs);return inputs
