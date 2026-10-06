"""CPU-only scaling experiment on frozen 16-PC ecological feature grids."""
import argparse,json,time,warnings,shutil,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans,HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import clustering_methods10 as prior

ROOT=Path('/home/masahiro/kics-zert2/runs/eco_whitening10_v1')
SCALES={'current':0.,'partial':.5,'full':1.}
ALGS=['kmeans','gmm','hdbscan']

def weights(r,alpha):
    relative=r/r.mean();safe=np.maximum(relative,1e-6)
    return safe**(-alpha/2)

def align(base,lab):
    aa=np.unique(base[base>=0]);bb=np.unique(lab[lab>=0]);mapping={}
    if len(aa) and len(bb):
        both=(base>=0)&(lab>=0);counts=np.zeros((len(aa),len(bb)),np.int64)
        np.add.at(counts,(np.searchsorted(aa,base[both]),np.searchsorted(bb,lab[both])),1)
        i,j=linear_sum_assignment(-counts)
        mapping={int(bb[y]):int(aa[x]) for x,y in zip(i,j) if counts[x,y]>0}
    nextid=int(aa.max()+1) if len(aa) else 0
    for value in bb:
        if int(value) not in mapping:mapping[int(value)]=nextid;nextid+=1
    return mapping

def fit(x,alg):
    extra={}
    with warnings.catch_warnings(record=True) as caught:
        if alg=='kmeans':
            model=KMeans(n_clusters=6,n_init=10,random_state=42).fit(x);lab=model.labels_
            extra={'iterations':int(model.n_iter_)}
        elif alg=='gmm':
            model=GaussianMixture(n_components=6,covariance_type='full',n_init=3,max_iter=300,tol=.001,reg_covar=1e-5,random_state=42).fit(x)
            prob=model.predict_proba(x);lab=prob.argmax(1)
            extra={'converged':bool(model.converged_),'iterations':int(model.n_iter_),'mean_max_posterior':float(prob.max(1).mean())}
        else:
            model=HDBSCAN(min_cluster_size=max(32,round(.005*len(x))),min_samples=15,cluster_selection_method='eom',metric='euclidean',n_jobs=2).fit(x)
            lab=model.labels_;extra={'min_cluster_size':max(32,round(.005*len(x))),'min_samples':15}
        extra['warnings']=[str(v.message) for v in caught]
    return lab.astype(np.int32),extra

def render(d,stem,labels,base,rgb):
    mapping=align(base,labels);n=max(mapping.values(),default=0)+1
    palette=np.random.default_rng(33).integers(25,240,(n,3),dtype=np.uint8)
    colors=np.full((*labels.shape,3),180,np.uint8)
    for label,col in mapping.items():colors[labels==label]=palette[col]
    hh,ww=rgb.shape[:2];pic=Image.fromarray(colors).resize((ww,hh),Image.Resampling.NEAREST)
    pic.save(d/f'{stem}_clusters.png')
    full=np.asarray(Image.fromarray(labels).resize((ww,hh),Image.Resampling.NEAREST));valid=full>=0
    overlay=rgb.copy();overlay[valid]=np.uint8(.55*rgb[valid]+.45*np.asarray(pic)[valid])
    Image.fromarray(overlay).save(d/f'{stem}_overlay.jpg',quality=94)
    edges=np.zeros(full.shape,bool);edges[1:]|=full[1:]!=full[:-1];edges[:,1:]|=full[:,1:]!=full[:,:-1]
    boundary=rgb.copy();boundary[edges]=[255,35,150]
    Image.fromarray(boundary).save(d/f'{stem}_boundary.jpg',quality=95)
    return mapping

def analyze(row):
    with threadpool_limits(limits=2):
        source=prior.ROOT/row['id'];d=ROOT/row['id'];d.mkdir(exist_ok=True)
        for name in ['input.jpg','source.json']:shutil.copy2(source/name,d/name)
        rgb=np.asarray(Image.open(source/'input.jpg'))
        cal=prior.old.ROOT/row['id']/'calibration.json';r=np.array(json.loads(cal.read_text())['explained_variance_ratio'],np.float64)
        assert r.shape==(16,) and np.all(r>0)
        record={'id':row['id'],'calibration_sha256':prior.p1.sha(cal),'explained_variance_ratio':r.tolist(),'retained_variance_fraction':float(r.sum()),'relative_variance':(r/r.mean()).tolist(),'variance_floor_activated':bool(np.any(r/r.mean()<1e-6)),'scales':{},'methods':{}}
        for name,a in SCALES.items():
            w=weights(r,a);v=r*w*w
            record['scales'][name]={'alpha':a,'multipliers':w.tolist(),'calibration_shares':(v/v.sum()).tolist()}
        np.testing.assert_allclose(r*weights(r,1)**2,np.full(16,r.mean()),rtol=1e-12)
        for method in prior.METHODS:
            shutil.copy2(source/f'{method}_pca.jpg',d/f'{method}_pca.jpg')
            f=source/f'{method}_features.npz';raw=np.load(f)['features'];gh,gw=raw.shape[:2];x=raw.reshape(-1,16)
            fhash=prior.p1.sha(f);v=np.var(x.astype(np.float64),axis=0,ddof=1)
            record['methods'][method]={'feature_sha256':fhash,'grid_shape':[gh,gw],'observed_variance':v.tolist(),'observed_shares':{}}
            for scale,alpha in SCALES.items():
                w=weights(r,alpha);scaled=np.ascontiguousarray(x*w.astype(np.float32));actual=np.var(scaled.astype(np.float64),axis=0,ddof=1)
                np.testing.assert_allclose(actual,v*w*w,rtol=2e-6,atol=1e-8)
                if alpha==0:np.testing.assert_array_equal(scaled,x)
                record['methods'][method]['observed_shares'][scale]=(actual/actual.sum()).tolist()
                for alg in ALGS:
                    stem=f'{method}_{alg}_{scale}';target=d/f'{stem}.json'
                    if target.exists():continue
                    base=np.load(source/f'{method}_{alg}_labels.npz')['labels']
                    if scale=='current':
                        lab=base.copy();prev=json.loads((source/f'{method}_{alg}.json').read_text());assert prev['feature_sha256']==fhash
                        extra={k:prev[k] for k in ['converged','iterations','mean_max_posterior','min_cluster_size','min_samples'] if k in prev};seconds=None
                    else:
                        start=time.perf_counter();flat,extra=fit(scaled,alg);seconds=time.perf_counter()-start;lab=flat.reshape(gh,gw)
                    assert lab.shape==base.shape
                    values,counts=np.unique(lab,return_counts=True);both=(base>=0)&(lab>=0)
                    mapping=render(d,stem,lab,base,rgb)
                    edge=np.zeros(lab.shape,bool);edge[1:]|=lab[1:]!=lab[:-1];edge[:,1:]|=lab[:,1:]!=lab[:,:-1]
                    out={'id':row['id'],'method':method,'algorithm':alg,'scale':scale,'alpha':alpha,'feature_sha256':fhash,'calibration_sha256':record['calibration_sha256'],'grid_shape':[gh,gw],'cluster_count':int((values>=0).sum()),'noise_fraction':float((lab<0).mean()),'area_fractions':{str(int(vv)):float(cc/lab.size) for vv,cc in zip(values,counts)},'ari_all_vs_current':float(adjusted_rand_score(base.ravel(),lab.ravel())),'ari_common_assigned':float(adjusted_rand_score(base[both],lab[both])) if both.sum()>1 else None,'common_assigned_fraction':float(both.mean()),'boundary_density':float(edge.mean()),'display_color_mapping':mapping,'seconds':seconds,'baseline_reused':scale=='current',**extra}
                    np.savez_compressed(d/f'{stem}_labels.npz',labels=lab)
                    prior.mb.save(target,out);print('DONE',row['id'],method,alg,scale,out['cluster_count'],round(out['noise_fraction'],3),flush=True)
        prior.mb.save(d/'variance.json',record)
        return row['id']

def native_control():
    d=ROOT/'moin_01';source=prior.old.ROOT/'moin_01';tr=np.load(source/'transform.npz')
    grid=prior.mb.project(np.load(source/'tokens_whole.npz')['z'],tr['basis'],tr['mean'])[0]
    r=np.array(json.loads((source/'calibration.json').read_text())['explained_variance_ratio']);base=np.load(prior.ROOT/'moin_01'/'whole_native_hdbscan_labels.npz')['labels'];rgb=np.asarray(Image.open(d/'input.jpg'))
    out={}
    with threadpool_limits(limits=2):
        for scale,a in SCALES.items():
            lab=base if a==0 else fit(np.ascontiguousarray(grid.reshape(-1,16)*weights(r,a).astype(np.float32)),'hdbscan')[0].reshape(grid.shape[:2])
            render(d,f'native_hdbscan_{scale}',lab,base,rgb)
            np.savez_compressed(d/f'native_hdbscan_{scale}_labels.npz',labels=lab)
            out[scale]={'cluster_count':int(len(np.unique(lab[lab>=0]))),'noise_fraction':float((lab<0).mean()),'grid_shape':list(lab.shape)}
    prior.mb.save(ROOT/'native_control.json',out)

def analyze_pc3(row):
    with threadpool_limits(limits=2):
        source=prior.ROOT/row['id'];d=ROOT/row['id'];d.mkdir(exist_ok=True)
        rgb=np.asarray(Image.open(source/'input.jpg'))
        for method in prior.METHODS:
            f=source/f'{method}_features.npz';raw=np.load(f)['features'];gh,gw=raw.shape[:2]
            x=np.ascontiguousarray(raw[:,:,:3].reshape(-1,3));assert np.isfinite(x).all()
            for alg in ALGS:
                stem=f'{method}_{alg}_pc3';target=d/f'{stem}.json'
                if target.exists():continue
                base=np.load(source/f'{method}_{alg}_labels.npz')['labels']
                start=time.perf_counter();flat,extra=fit(x,alg);seconds=time.perf_counter()-start;lab=flat.reshape(gh,gw)
                values,counts=np.unique(lab,return_counts=True);both=(base>=0)&(lab>=0)
                mapping=render(d,stem,lab,base,rgb)
                edges=np.zeros(lab.shape,bool);edges[1:]|=lab[1:]!=lab[:-1];edges[:,1:]|=lab[:,1:]!=lab[:,:-1]
                out={'id':row['id'],'method':method,'algorithm':alg,'scale':'pc3','alpha':None,'retained_components':[1,2,3],'feature_selection':'First three raw PCA scores; no whitening or display-channel rescaling.','feature_sha256':prior.p1.sha(f),'grid_shape':[gh,gw],'cluster_count':int((values>=0).sum()),'noise_fraction':float((lab<0).mean()),'area_fractions':{str(int(vv)):float(cc/lab.size) for vv,cc in zip(values,counts)},'ari_all_vs_current':float(adjusted_rand_score(base.ravel(),lab.ravel())),'ari_common_assigned':float(adjusted_rand_score(base[both],lab[both])) if both.sum()>1 else None,'common_assigned_fraction':float(both.mean()),'boundary_density':float(edges.mean()),'display_color_mapping':mapping,'seconds':seconds,'baseline_reused':False,**extra}
                np.savez_compressed(d/f'{stem}_labels.npz',labels=lab);prior.mb.save(target,out)
                print('PC3_DONE',row['id'],method,alg,out['cluster_count'],round(out['noise_fraction'],3),flush=True)
        return row['id']

def pc3_native_control():
    d=ROOT/'moin_01';source=prior.old.ROOT/'moin_01';tr=np.load(source/'transform.npz')
    grid=prior.mb.project(np.load(source/'tokens_whole.npz')['z'],tr['basis'],tr['mean'])[0][:,:,:3]
    base=np.load(prior.ROOT/'moin_01'/'whole_native_hdbscan_labels.npz')['labels'];rgb=np.asarray(Image.open(d/'input.jpg'))
    with threadpool_limits(limits=2):lab=fit(np.ascontiguousarray(grid.reshape(-1,3)),'hdbscan')[0].reshape(grid.shape[:2])
    render(d,'native_hdbscan_pc3',lab,base,rgb);np.savez_compressed(d/'native_hdbscan_pc3_labels.npz',labels=lab)
    out=json.loads((ROOT/'native_control.json').read_text());out['pc3']={'cluster_count':int(len(np.unique(lab[lab>=0]))),'noise_fraction':float((lab<0).mean()),'grid_shape':list(lab.shape)}
    prior.mb.save(ROOT/'native_control.json',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pc3-only',action='store_true');args=ap.parse_args()
    ROOT.mkdir(exist_ok=True);rows=prior.selection();shutil.copy2(prior.ROOT/'selection.json',ROOT/'selection.json')
    if args.pc3_only:
        assert (ROOT/'native_control.json').exists(),'Run the scaling phase before PC3'
        protocol=json.loads((ROOT/'protocol.json').read_text());protocol['pc3_extension']={'script_sha256':prior.p1.sha(Path(__file__)),'condition':'Joint clustering on raw PC1,PC2,PC3 only; same grid, model settings and seeds; no PCA refit or RGB display rescaling.'};prior.mb.save(ROOT/'protocol.json',protocol)
        with ProcessPoolExecutor(max_workers=3) as pool:
            for done in pool.map(analyze_pc3,rows):print('PC3_IMAGE_COMPLETE',done,flush=True)
        pc3_native_control();print('PC3_COMPLETE',flush=True);raise SystemExit(0)
    prior.mb.save(ROOT/'protocol.json',{'script_sha256':prior.p1.sha(Path(__file__)),'source_run':str(prior.ROOT),'images':10,'methods':prior.METHODS,'algorithms':ALGS,'alpha':SCALES,'scaling':'r = saved PCA calibration explained_variance_ratio; v = r / mean(r); coordinate multiplier = max(v,1e-6)^(-alpha/2). Current alpha0; partial0.5; full1. Relative whitening: full makes calibration variances equal up to one common scale, not necessarily unit variance. Calibration is shared across feature treatments within each image. No per-treatment variance fitting, PCA refitting or DINO rerun.','calibration':'Original PCA basis was calibrated on balanced token samples across six feature treatments, including whole-fourflip and Local40. Only whole,Local5,10,20 are evaluated. Saved ratios suffice for relative whitening; absolute eigenvalues were not saved.','parameters':'Same KMeans K6,n_init10,seed42; GMM six full covariances,n_init3,max_iter300,tol.001,reg_covar1e-5,seed42; HDBSCAN Euclidean,EOM,min_cluster_size=max(32,round(.005*N)),min_samples15.','baseline':'Original labels reused exactly, validated against source feature hashes.','display':'Colors matched one-to-one to baseline by maximum assigned-pixel overlap; unmatched groups receive extra colors. Raw labels unchanged. Gray is unassigned. PCA RGB reference unchanged across scaling.','interpretation':'Agreement to baseline measures change, not accuracy. Common-assigned ARI excludes noise and is accompanied by coverage. Full covariance GMM is theoretically affine-equivariant at corresponding solutions without fixed regularization; initialization and fixed reg_covar can create practical changes. Fixed spatial grid, including interpolated whole tokens, remains a sensitivity. Native128 grid HDBSCAN control for moin01 repeated at all three scalings.','execution':'CPU-only, three image workers, two numeric threads each. No new energy or runtime comparison; baseline cached and concurrent timings non-comparable.'})
    with ProcessPoolExecutor(max_workers=3) as pool:
        for done in pool.map(analyze,rows):print('IMAGE_COMPLETE',done,flush=True)
    native_control();print('COMPLETE',flush=True)
