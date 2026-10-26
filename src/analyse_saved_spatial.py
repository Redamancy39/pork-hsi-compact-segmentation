"""CPU-only, read saved class IDs/logits; never import a model or torch."""
from pathlib import Path
import argparse, csv, hashlib, json, time
import numpy as np
from scipy.ndimage import maximum_filter

CONFIG=None
OUT=None
ROOT=None
def configure(path,out):
    global CONFIG,OUT,ROOT
    path=Path(path).resolve();ROOT=path.parent;CONFIG=json.loads(path.read_text('utf-8'));OUT=Path(out).resolve()
    for key in ['members','class_ids','expected_source_counts']:
        CONFIG[key]=str((ROOT/CONFIG[key]).resolve())
    for r in CONFIG['predictions']:r['logits']=str((ROOT/r['logits']).resolve())
    if {(r['family'],int(r['seed'])) for r in CONFIG['predictions']}!={(f,s) for f in FAMILIES for s in SEEDS}:raise ValueError('All four families and four seeds required')

SEEDS = (42, 43, 44, 45)
FAMILIES = ('B0', 'B1', 'T', 'B1_ALL_D1')
DIST = ('d1', 'd2_3', 'd4_plus', 'counterpart_absent')
EDGE = ('outer_0_1', 'middle_2_3', 'inner_4_plus')

def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def jread(p): return json.loads(p.read_text('utf-8'))
def percentage(n, d): return 100.0 * n / d if d else 'NA'

def bands(counterpart):
    # No cross-patch filtering and no imaginary outside-crop support.
    near1 = maximum_filter(counterpart, size=(1, 3, 3), mode='constant', cval=0)
    near3 = maximum_filter(counterpart, size=(1, 7, 7), mode='constant', cval=0)
    present = counterpart.any(axis=(1, 2))[:, None, None]
    b = np.where(near1, 0, np.where(near3, 1, np.where(present, 2, 3))).astype(np.uint8)
    return b

def prediction(f,s):
    r=next(r for r in CONFIG['predictions'] if r['family']==f and int(r['seed'])==s)
    return Path(r['logits']),r

def main():
    start = time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    members = read_csv(Path(CONFIG['members'])); ids = np.load(Path(CONFIG['class_ids']), allow_pickle=False)
    assert ids.shape == (2048, 20, 16)
    mh = sha(Path(CONFIG['members'])); ih = sha(Path(CONFIG['class_ids']))
    qc = CONFIG
    assert mh == qc['manifest_sha256'] and ih == qc['ids_sha256']
    parent = np.array([int(r['parent'].rsplit('_', 1)[1]) for r in members])
    assert set(parent) == {11,21,35,62} and all((parent == s).sum() == 512 for s in set(parent))
    fg = (ids >= 0) & (ids <= 9); bg = (ids >= 10) & (ids <= 12)
    assert fg.sum() == 12825 and bg.sum() == 642535 and np.all(fg | bg)
    regions = {s: np.flatnonzero(parent == s) for s in (11,21,35,62)}
    regions['ALL'] = np.arange(2048)
    dfg = bands(bg); dbg = bands(fg)
    yy, xx = np.indices((20,16)); e = np.minimum.reduce([yy,xx,19-yy,15-xx])
    eb = np.where(e <= 1, 0, np.where(e <= 3, 1, 2))
    edge = np.broadcast_to(eb, ids.shape)
    tissue_d = {'meat': bands(ids == 11), 'fat': bands(ids == 10)}
    expected = {(r['family'],int(r['seed']),int(r['parent'].rsplit('_',1)[1])): r for r in read_csv(Path(CONFIG['expected_source_counts']))}
    masks = {}; identities = []; reproduced = []
    for f in FAMILIES:
        for seed in SEEDS:
            p, receipt = prediction(f,seed)
            assert receipt['manifest_sha256'] == mh and receipt['logits_sha256'] == sha(p)
            logits = np.load(p,allow_pickle=False)
            assert logits.shape == ids.shape and np.isfinite(logits).all()
            pred = logits >= 0; masks[f,seed] = pred
            identities.append(dict(family=f,seed=seed,prediction_path=str(p),logits_sha256=receipt['logits_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],manifest_sha256=mh))
            for source, ix in regions.items():
                if source == 'ALL': continue
                a=ids[ix]; pr=pred[ix]; g=fg[ix]; b=bg[ix]
                got=dict(TP=int((pr&g).sum()),FP=int((pr&b).sum()),FN=int((~pr&g).sum()),TN=int((~pr&b).sum()))
                for tissue,code in [('meat',10),('fat',11),('cb',12)]:
                    got[tissue+'_bg']=int((a==code).sum());got[tissue+'_fp']=int(((a==code)&pr).sum())
                exp=expected[f,seed,source]
                assert all(got[k] == int(exp[k]) for k in got), (f,seed,source,got)
                dice=2*got['TP']/(2*got['TP']+got['FP']+got['FN'])
                assert abs(dice-float(exp['dice']))<1e-14
                reproduced.append(dict(family=f,seed=seed,source=source,**got,dice=dice,all_original_counts_equal=True))

    transitions=[]; tissue_rows=[]; error_rows=[]
    for seed in SEEDS:
        a=masks['B1',seed]; b=masks['B1_ALL_D1',seed]
        tr={'recovered_FG':fg&a&~b, 'new_FN':fg&~a&b, 'removed_FP':bg&~a&b, 'new_FP':bg&a&~b}
        for source,ix in regions.items():
            for domain, support, d in [('FG',fg,dfg),('BG',bg,dbg)]:
                keys=('recovered_FG','new_FN') if domain=='FG' else ('removed_FP','new_FP')
                for scheme,categories,idxmap in [('opposite_class_distance',DIST,d),('crop_edge',EDGE,edge)]:
                    for k,category in enumerate(categories):
                        sel=support[ix] & (idxmap[ix]==k); n=int(sel.sum())
                        vals={name:int((tr[name][ix]&sel).sum()) for name in keys}
                        transitions.append(dict(source=source,seed=seed,domain=domain,scheme=scheme,stratum=category,support=n,
                            recovered_FG=vals.get('recovered_FG',0),new_FN=vals.get('new_FN',0),removed_FP=vals.get('removed_FP',0),new_FP=vals.get('new_FP',0),
                            rate_first_percent=percentage(vals[keys[0]],n),rate_second_percent=percentage(vals[keys[1]],n)))
                if domain=='BG':
                    for k,category in enumerate(DIST):
                        for q,ename in enumerate(EDGE):
                            sel=support[ix]&(d[ix]==k)&(edge[ix]==q);n=int(sel.sum())
                            vals={name:int((tr[name][ix]&sel).sum()) for name in keys}
                            transitions.append(dict(source=source,seed=seed,domain=domain,scheme='BG_distance_x_crop_edge',stratum=category+'__'+ename,support=n,
                                recovered_FG=0,new_FN=0,**vals,rate_first_percent=percentage(vals[keys[0]],n),rate_second_percent=percentage(vals[keys[1]],n)))
            for f in FAMILIES:
                pr=masks[f,seed][ix]
                for tissue,code in [('meat',10),('fat',11)]:
                    for k,category in enumerate(DIST):
                        sel=(ids[ix]==code)&(tissue_d[tissue][ix]==k);n=int(sel.sum());fp=int((sel&pr).sum())
                        tissue_rows.append(dict(source=source,family=f,seed=seed,tissue=tissue,opposite_tissue_distance=category,support=n,FP=fp,FP_percent=percentage(fp,n)))
                for domain,support,d in [('FG',fg,dfg),('BG',bg,dbg)]:
                    for k,category in enumerate(DIST):
                        sel=support[ix]&(d[ix]==k);n=int(sel.sum());err=int((sel&(~pr if domain=='FG' else pr)).sum())
                        error_rows.append(dict(source=source,family=f,seed=seed,domain=domain,stratum=category,support=n,error_count=err,error_percent=percentage(err,n)))
        # All transitions reconcile the existing full-patch counts.
        assert int(tr['recovered_FG'].sum())-int(tr['new_FN'].sum())==int((a&fg).sum())-int((b&fg).sum())
        assert int(tr['new_FP'].sum())-int(tr['removed_FP'].sum())==int((a&bg).sum())-int((b&bg).sum())

    # Independent distance route: source62, seed42, explicit all-coordinate distances.
    brute_checks=0
    ix=regions[62]
    for i in ix:
        for counter,cached in [(bg,dfg),(fg,dbg),(ids==11,tissue_d['meat']),(ids==10,tissue_d['fat'])]:
            pts=np.argwhere(counter[i]); answer=np.full((20,16),3,dtype=np.uint8)
            if len(pts):
                d=np.abs(np.indices((20,16)).transpose(1,2,0)[:,:,None,:]-pts[None,None,:,:]).max(axis=-1).min(axis=-1)
                answer=np.where(d<=1,0,np.where(d<=3,1,2)).astype(np.uint8)
            assert np.array_equal(answer,cached[i]), (i,'distance')
            brute_checks+=1
    # Independently sum source62 seed42 from pixel tuples, without vector masks.
    a=masks['B1',42];b=masks['B1_ALL_D1',42]
    counts={'recovered_FG':0,'new_FN':0,'removed_FP':0,'new_FP':0}
    tissue_brute={f:[0,0,0,0] for f in FAMILIES}
    for i in ix:
        for y in range(20):
            for x in range(16):
                code=int(ids[i,y,x]);ap=bool(a[i,y,x]);bp=bool(b[i,y,x])
                if code<10:
                    counts['recovered_FG']+=int(ap and not bp);counts['new_FN']+=int(bp and not ap)
                else:
                    counts['removed_FP']+=int(bp and not ap);counts['new_FP']+=int(ap and not bp)
                if code==10:
                    for f in FAMILIES: tissue_brute[f][int(tissue_d['meat'][i,y,x])]+=int(masks[f,42][i,y,x])
    for key,v in counts.items():
        assert v==sum(r[key] for r in transitions if r['source']==62 and r['seed']==42 and r['scheme']=='opposite_class_distance')
    for f in FAMILIES:
        assert tissue_brute[f]==[r['FP'] for r in tissue_rows if r['source']==62 and r['seed']==42 and r['family']==f and r['tissue']=='meat']

    write_csv(OUT/'spatial_transitions.csv',transitions)
    write_csv(OUT/'spatial_tissue_counts.csv',tissue_rows)
    write_csv(OUT/'spatial_error_counts.csv',error_rows)
    write_csv(OUT/'spatial_original_count_check.csv',reproduced)
    write_csv(OUT/'spatial_input_identities.csv',identities)
    result=dict(manifest_sha256=mh,ids_sha256=ih,patches=2048,foreground=12825,background=642535,ignore=0,
        original_source_rows_exact=len(reproduced),independent_bruteforce_distance_maps=brute_checks,
        independent_source62_seed42_transitions=counts,independent_source62_seed42_meat_FP=tissue_brute,
        analysis_code_sha256=sha(Path(__file__)),
        new_model_forwards=0,optimizer_updates=0,new_timing=0,downloads=0,
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();configure(args.config,args.out);main()

