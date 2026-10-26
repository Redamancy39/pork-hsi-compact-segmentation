"""Official Cuvis readback, fixed source QC, FIT-only spectra/PCA and caches."""
from .common import *
import numpy as np,sys,gc,ast,argparse
from collections import defaultdict
from types import SimpleNamespace
from typing import Iterable
from skimage.draw import polygon2mask

H,W,C=1000,1080,61
WAIT_FOR_DOWNLOAD=False
DOWNLOAD_WAIT_SECONDS=0.
def official_mask(coco,index):
    labels=np.zeros((H,W),dtype=np.uint8)
    for a in coco['annotations']:
        if int(a['image_id'])!=index:continue
        assert int(a['category_id']) in range(1,8)
        if not isinstance(a['segmentation'],list) or not a['segmentation']:raise RuntimeError('Missing polygon')
        for points in a['segmentation']:
            xy=np.asarray(points,dtype=np.float32).reshape(-1,2)
            if not np.isfinite(xy).all() or (xy<0).any() or (xy[:,0]>W).any() or (xy[:,1]>H).any():raise RuntimeError('Invalid polygon canvas')
            labels[polygon2mask((H,W),xy[:,[1,0]])]=a['category_id']
    return labels

def generate_coordinates():
    p=ROOT/'plans/crop_coordinates.csv'
    if p.exists():return read_csv(p)
    groups=defaultdict(list)
    for r in members('FIT_PROPOSED'):groups[r['group']].append(r)
    assert len(groups)==85 and all(len(v)==4 for v in groups.values())
    order=sorted(groups,key=lambda g:seeded_int('quota_'+g));extra=set(order[:8192%85]);coords=[]
    for group,rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda r:(r['source'],int(r['index'])))
        coco=read_json(METADATA/Path(rr[0]['source']).with_suffix('.json'))
        mask=official_mask(coco,int(rr[0]['index']));fg=np.flatnonzero(mask>0)
        rng=np.random.default_rng(seeded_int('FIT_CROPS_'+group));n=8192//85+(group in extra)
        for j in range(n):
            r=rr[j%4];positive=bool(len(fg) and (j//4)%2==0)
            if positive:
                pix=int(rng.choice(fg));yy,xx=divmod(pix,W)
                y=int(np.clip(yy-int(rng.integers(0,20)),0,H-20));x=int(np.clip(xx-int(rng.integers(0,16)),0,W-16))
            else:
                for _ in range(10000):
                    y=int(rng.integers(0,H-20+1));x=int(rng.integers(0,W-16+1))
                    if not mask[y:y+20,x:x+16].any():break
                else:raise RuntimeError('Pure-background pool unsupported')
            crop=mask[y:y+20,x:x+16]
            assert bool(crop.any())==positive
            coords.append(dict(patch_id=len(coords),source=r['source'],index=r['index'],group=group,y=y,x=x,
                               pool='foreground_intersection' if positive else 'pure_background',FG_pixels=int(np.count_nonzero(crop)),
                               classes=';'.join(map(str,np.unique(crop[crop>0]).tolist()))))
    assert len(coords)==8192;write_csv(p,coords)
    rows=members('FIT_PROPOSED');points=np.empty((len(rows),1024,2),dtype=np.int16)
    for i,r in enumerate(rows):
        rng=np.random.default_rng(seeded_int('PCA_COORDS_'+r['source']+'/'+r['index']));ix=rng.choice(H*W,1024,replace=False)
        points[i,:,0]=ix//W;points[i,:,1]=ix%W
    np.save(ROOT/'plans/FIT_spectrum_yx.npy',points)
    atomic_json(ROOT/'plans/COORDINATE_LOCK.json',dict(created_utc=utc(),crops_sha256=sha(p),spectra_coordinates_sha256=sha(ROOT/'plans/FIT_spectrum_yx.npy'),
                 purpose='FIT only; uniform per-frame spectra; group-balanced crop quota; no DEV/holdout mining'))
    return coords

_dll_handles=[]
_reader_instance=None
def init_reader():
    global _reader_instance
    if _reader_instance is not None:return _reader_instance
    temp=ROOT/'runtime_temp';temp.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp)
    if SETTINGS.get('cuvis_bin'):
        sdk=Path(SETTINGS['cuvis_bin']);os.environ['CUVIS']=str(sdk)
        os.environ['PATH']=str(sdk)+os.pathsep+os.environ.get('PATH','')
        if os.name=='nt':_dll_handles.append(os.add_dll_directory(str(sdk)))
    if os.name=='nt':
        import importlib.util
        spec=importlib.util.find_spec('torch')
        if spec and spec.origin:
            libs=Path(spec.origin).parent/'lib'
            if libs.exists():_dll_handles.append(os.add_dll_directory(str(libs)))
    import cuvis
    settings=ROOT/'reader_settings';settings.mkdir(exist_ok=True)
    cuvis.SdkSettings(force_gpu_mode='host',processing_thread_count=4).save(settings/'cuvis.settings')
    cuvis.init(settings,global_loglevel='warning');_reader_instance=cuvis
    return cuvis

def iterate_frames(rows,save_qc=True):
    global DOWNLOAD_WAIT_SECONDS
    cuvis=init_reader();qc_records=[];allby=defaultdict(list)
    for r in rows:allby[r['source']].append(r)
    qc_path=ROOT/'qc'/f'{rows[0]["role"]}_source_qc.json'
    for source,rr in allby.items():
        waiting=time.perf_counter()
        while not container_path(source).exists():
            if not WAIT_FOR_DOWNLOAD:raise RuntimeError('Required verified container unavailable: '+source)
            progress('PREPARATION_WAITING_FOR_DOWNLOAD',source=source)
            error=ROOT/'download_fit_error.json'
            pidfile=ROOT/'download_fit.pid'
            if error.exists() and pidfile.exists():
                import psutil
                if not psutil.pid_exists(int(pidfile.read_text().strip())):raise RuntimeError('Downloader stopped before required container')
            time.sleep(10)
        DOWNLOAD_WAIT_SECONDS+=time.perf_counter()-waiting
        started=time.perf_counter();session=cuvis.SessionFile(container_path(source));ctx=cuvis.ProcessingContext(session)
        if not ctx.has_reference(cuvis.ReferenceType.Dark) or not ctx.has_reference(cuvis.ReferenceType.White):raise RuntimeError('Missing source dark/white reference')
        ctx.processing_mode=cuvis.ProcessingMode.Reflectance
        coco=read_json(METADATA/Path(source).with_suffix('.json'));images={int(x['id']):x for x in coco['images']}
        assert len(session)==len(images)
        for r in rr:
            t=time.perf_counter();idx=int(r['index']);m=session.get_measurement(idx)
            if int(m.name.rsplit('_',1)[1])!=int(images[idx]['camera_frame_num']):raise RuntimeError('Camera frame identity mismatch')
            ctx.apply(m);arr=m.data['cube'].array;wl=np.asarray(m.data['cube'].wavelength)
            if arr.shape!=(H,W,C) or arr.dtype!=np.uint16 or not np.array_equal(wl,np.arange(430,911,8)):raise RuntimeError('Unexpected spectral schema')
            if not np.any(arr):raise RuntimeError('All-zero source cube')
            labels=official_mask(coco,idx)
            if bool(labels.any())!=(r['positive']=='True'):raise RuntimeError('Metadata/label support mismatch')
            if r['documented_normal_recording']=='True' and labels.any():raise RuntimeError('Documented normal recording contains annotation')
            rec=dict(source=source,index=idx,group=r['group'],role=r['role'],name=m.name,capture_time=str(m.capture_time),
                     serial=m.serial_number,shape=list(arr.shape),dtype=str(arr.dtype),reference_Dark=True,reference_White=True,
                     mode=str(m.processing_mode),min=int(arr.min()),max=int(arr.max()),FG=int(np.count_nonzero(labels)),BG=int(np.count_nonzero(labels==0)),
                     classes=np.unique(labels[labels>0]).tolist(),read_reflectance_seconds=time.perf_counter()-t,
                     normal_status='documented_normal' if r['documented_normal_recording']=='True' else ('published_zero_region_visual_not_certified' if not labels.any() else 'positive'))
            qc_records.append(rec)
            if save_qc:save_representative(arr,labels,r,wl)
            atomic_json(qc_path,dict(updated_utc=utc(),frames=qc_records,visual_status='source-derived representatives saved; no whole-dataset manual certification'))
            yield r,arr,labels,rec
            del arr,m,labels;gc.collect()
        del ctx,session;gc.collect()
    # Keep the native runtime alive for this worker. Explicit shutdown while
    # caller-held measurement views remain live caused a cuvis.dll AV. The
    # worker process owns/reclaims the runtime after the complete stage.

def save_representative(arr,labels,r,wl):
    # Deterministic first FIT example per class / first normal; never score-selected.
    if r['role']=='DEV_PROPOSED':return
    classes=np.unique(labels[labels>0]).tolist() or [0]
    prefix='holdout_' if r['role']=='DAY4_HOLDOUT_PROPOSED' else 'fit_'
    for c in classes:
        dest=ROOT/'qc'/f'{prefix}class{c}.npz'
        if dest.exists():continue
        rgb=arr[:,:,[int(np.argmin(abs(wl-x))) for x in [650,550,450]]].copy()
        np.savez_compressed(dest,rgb=rgb,labels=labels,source=r['source'],index=int(r['index']),category=c)

def fit_preprocessing(spectra):
    X=np.asarray(spectra,dtype=np.float64)
    mean=X.mean(0);scale=X.std(0);scale=np.maximum(scale,1e-8)
    X=(X-mean)/scale;pca_mean=X.mean(0);X-=pca_mean
    cov=X.T@X/(len(X)-1);vals,vec=np.linalg.eigh(cov);ix=np.argsort(vals)[::-1];vals=vals[ix];components=vec[:,ix[:16]].T
    for row in components:
        if row[np.argmax(abs(row))]<0:row*=-1
    return dict(mean=mean.astype('float32'),scale=scale.astype('float32'),pca_mean=pca_mean.astype('float32'),
                components=components.astype('float32'),explained_variance=vals[:16].astype('float32'),explained_variance_ratio=(vals[:16]/vals.sum()).astype('float64'),
                spectrum_count=np.array(len(X),dtype=np.int64))

def project(a,prep):
    f=np.asarray(a,dtype=np.float32)
    f=(f-prep['mean'])/prep['scale'];f=f@prep['components'].T-(prep['components']@prep['pca_mean'])
    if not np.isfinite(f).all():raise RuntimeError('Nonfinite projected features')
    return f

def prepare_fit_dev(recover_fit_cache=False):
    lock_inputs()
    if (ROOT/'PREPARATION_COMPLETE.json').exists():return
    require_data(['FIT','DEV'])
    coords=generate_coordinates();fit_rows=members('FIT_PROPOSED');points=np.load(ROOT/'plans/FIT_spectrum_yx.npy')
    start=time.perf_counter()
    if recover_fit_cache:
        state=read_json(ROOT/'preparation_progress.json')
        qc=read_json(ROOT/'qc/FIT_PROPOSED_source_qc.json')['frames']
        assert state['stage']=='FIT_READBACK' and state['completed']==340 and len(qc)==340
        assert [(r['source'],int(r['index'])) for r in fit_rows]==[(r['source'],int(r['index'])) for r in qc]
        y=np.load(ROOT/'cache/train_labels.npy',mmap_mode='r')
        spectra=np.load(ROOT/'cache/FIT_spectra.npy',mmap_mode='r')
        x=np.load(ROOT/'cache/train_features.npy',mmap_mode='r')
        assert x.shape==(8192,16,20,16) and x.dtype==np.float32 and y.shape==(8192,20,16) and spectra.shape==(348160,61)
        assert np.isfinite(spectra).all() and np.all(np.std(spectra,axis=0)>0)
        assert np.isin(y,[0,1]).all()
        assert np.array_equal(y.sum((1,2)),np.array([int(c['FG_pixels']) for c in coords]))
        assert not (ROOT/'cache/train_raw_crops.npy').exists() # deleted only after all projection chunks flushed
        assert np.isfinite(x).all() and np.all(np.any(x!=0,axis=(1,2,3)))
        prep=dict(np.load(ROOT/'preprocessing.npz'))
        lock=read_json(ROOT/'locks/PREPROCESSING_LOCK.json');assert lock['sha256']==sha(ROOT/'preprocessing.npz')
        refit=fit_preprocessing(spectra)
        assert all(np.array_equal(prep[k],refit[k]) for k in prep)
        atomic_json(ROOT/'logs/FIT_CACHE_RECOVERY.json',dict(verified_utc=utc(),frames=340,crops=8192,labels_match_all_saved_coordinates=True,
            spectra_sha256=sha(ROOT/'cache/FIT_spectra.npy'),features_sha256=sha(ROOT/'cache/train_features.npy'),frozen_preprocessing_exact_reproduction=True,
            reason='Native Cuvis shutdown/re-init AV after complete FIT projection; reuse complete flushed cache in fresh worker, same frozen PCA'))
        del x,y,spectra,refit;gc.collect()
    else:
        raw=np.lib.format.open_memmap(ROOT/'cache/train_raw_crops.npy',mode='w+',dtype=np.uint16,shape=(8192,20,16,61))
        y=np.lib.format.open_memmap(ROOT/'cache/train_labels.npy',mode='w+',dtype=np.uint8,shape=(8192,20,16))
        spectra=np.lib.format.open_memmap(ROOT/'cache/FIT_spectra.npy',mode='w+',dtype=np.float32,shape=(340*1024,61))
        byframe=defaultdict(list)
        for c in coords:byframe[(c['source'],int(c['index']))].append(c)
        for i,(r,a,labels,rec) in enumerate(iterate_frames(fit_rows)):
            assert (r['source'],r['index'])==(fit_rows[i]['source'],fit_rows[i]['index'])
            yy,xx=points[i].T;spectra[i*1024:(i+1)*1024]=a[yy,xx]
            for c in byframe[(r['source'],int(r['index']))]:
                j=int(c['patch_id']);cy,cx=int(c['y']),int(c['x']);raw[j]=a[cy:cy+20,cx:cx+16];y[j]=(labels[cy:cy+20,cx:cx+16]>0)
                if int(y[j].sum())!=int(c['FG_pixels']):raise RuntimeError('Crop-label identity mismatch')
            atomic_json(ROOT/'preparation_progress.json',dict(stage='FIT_READBACK',completed=i+1,total=340,updated_utc=utc(),elapsed_seconds=time.perf_counter()-start))
            if i%10==0:progress('PREPARING_FIT',frame=i+1,total=340)
        raw.flush();y.flush();spectra.flush();prep=fit_preprocessing(spectra)
        np.savez(ROOT/'preprocessing.npz',**prep)
        atomic_json(ROOT/'locks/PREPROCESSING_LOCK.json',dict(created_utc=utc(),sha256=sha(ROOT/'preprocessing.npz'),FIT_frames=340,FIT_spectrum_count=340*1024,
                    member_sha256=sha(ROOT/'members.csv'),coordinate_lock=read_json(ROOT/'plans/COORDINATE_LOCK.json'),DEV_spectra_used=0,holdout_spectra_used=0,
                    explained_variance_ratio_16=float(np.sum(prep['explained_variance_ratio'])),algorithm='FIT population standardisation; sample covariance eigendecomposition; descending16, max-loading-positive sign'))
        x=np.lib.format.open_memmap(ROOT/'cache/train_features.npy',mode='w+',dtype=np.float32,shape=(8192,16,20,16))
        for i in range(0,8192,256):x[i:i+256]=project(raw[i:i+256],prep).transpose(0,3,1,2)
        x.flush();del x,raw,y,spectra;gc.collect()
        # Explicitly named transient derived crop cache only; original cu3s remains intact.
        (ROOT/'cache/train_raw_crops.npy').unlink()
    devdir=ROOT/'cache/dev';devdir.mkdir(exist_ok=True)
    for i,(r,a,labels,rec) in enumerate(iterate_frames(members('DEV_PROPOSED'),save_qc=False)):
        key=frame_key(r);features=project(a,prep).transpose(2,0,1)
        np.save(devdir/f'{key}_pca.npy',features);np.save(devdir/f'{key}_labels.npy',labels)
        atomic_json(ROOT/'preparation_progress.json',dict(stage='DEV_TRANSFORM_ONLY',completed=i+1,total=60,updated_utc=utc(),elapsed_seconds=time.perf_counter()-start))
        if i%5==0:progress('PREPARING_DEV',frame=i+1,total=60)
    elapsed=time.perf_counter()-start;active_elapsed=elapsed-DOWNLOAD_WAIT_SECONDS
    if active_elapsed>7200:raise RuntimeError('CPU preparation budget exceeded')
    atomic_json(ROOT/'PREPARATION_COMPLETE.json',dict(completed_utc=utc(),FIT_frames=340,DEV_frames=60,holdout_HSI_frames_read=0,crop_count=8192,
                 preprocessing_sha256=sha(ROOT/'preprocessing.npz'),seconds=elapsed,active_CPU_stage_seconds=active_elapsed,network_wait_seconds=DOWNLOAD_WAIT_SECONDS,train_features_sha256=sha(ROOT/'cache/train_features.npy'),
                 train_labels_sha256=sha(ROOT/'cache/train_labels.npy')))
    progress('PREPARATION_COMPLETE',seconds=elapsed)
