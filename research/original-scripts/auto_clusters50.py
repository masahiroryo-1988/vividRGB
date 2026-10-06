"""Automatic K and coverage-oriented HDBSCAN on frozen ecological PCA grids.

No labels or RGB edges enter model selection. Spatial coordinates only partition
resampling blocks; they are never included in clustering distances.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, json, time, warnings, shutil, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._hdbscan.hdbscan import tree_to_labels
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
import auto_clusters50_extract as extraction
import pca_whitening10 as visual

ROOT=extraction.ROOT
METHODS=['whole','s05','s10','s20']; DIMS=[3,16]; KS=list(range(3,13))

def save(path,data):
    p=Path(path);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def partitions(shape):
    h,w=shape;y,x=np.indices(shape)
    blocks=(np.minimum(y*8//h,7)*8+np.minimum(x*8//w,7)).ravel()
    rng=np.random.default_rng(240103)
    order=rng.permutation(64);fold=np.empty(64,int);fold[order]=np.arange(64)%3
    result=[]
    for r in range(3):
        train=np.flatnonzero(fold[blocks]!=r);test=np.flatnonzero(fold[blocks]==r)
        # All K candidates share these exact rows and seeds.
        result.append((rng.choice(train,min(3072,len(train)),False),rng.choice(test,min(1024,len(test)),False)))
    final=rng.choice(h*w,min(8192,h*w),False)
    return result,final

def fit_model(x,alg,k,seed,final=False):
    if alg=='kmeans':return KMeans(n_clusters=k,n_init=10 if final else 3,random_state=seed).fit(x)
    model=GaussianMixture(n_components=k,covariance_type='full',n_init=3 if final else 2,max_iter=300,tol=.001,reg_covar=1e-5,random_state=seed).fit(x)
    if not model.converged_:
        model.set_params(max_iter=600,warm_start=True,n_init=1);model.fit(x)
    return model

def summarize(lab):
    values,counts=np.unique(lab,return_counts=True)
    return {'cluster_count':int((values>=0).sum()),'noise_fraction':float((lab<0).mean()),'area_fractions':{str(int(v)):float(c/lab.size) for v,c in zip(values,counts)}}

def render_result(d,stem,lab,rgb,meta,base=None):
    np.savez_compressed(d/f'{stem}_labels.npz',labels=lab)
    visual.render(d,stem,lab,lab if base is None else base,rgb)
    meta.update(summarize(lab));save(d/f'{stem}.json',meta)

def automatic(x,shape,alg,d,stem,rgb,common):
    if (d/f'{stem}.json').exists():return
    start=time.perf_counter();splits,final_idx=partitions(shape)
    score=np.full((len(KS),3),np.nan);candidates=[];prediction={}
    probe=np.random.default_rng(53).choice(len(x),min(2048,len(x)),False)
    for i,k in enumerate(KS):
        rec={'k':k,'scores':[],'converged':[],'training_bic':[]}
        for fold,(train,test) in enumerate(splits):
            with warnings.catch_warnings(record=True) as caught:
                model=fit_model(x[train],alg,k,42+fold)
            converged=bool(getattr(model,'converged_',True));rec['converged'].append(converged)
            pred=model.predict(x[test]);unique=len(np.unique(pred))
            s=float(silhouette_score(x[test],pred)) if alg=='kmeans' and 1<unique<len(test) else (float(model.score(x[test])) if alg=='gmm' else None)
            if not converged:s=None
            rec['scores'].append(s)
            if s is not None:score[i,fold]=s
            if alg=='gmm':rec['training_bic'].append(float(model.bic(x[train])))
            prediction[k,fold]=model.predict(x[probe])
        rec['mean']=float(score[i].mean()) if np.isfinite(score[i]).all() else None
        rec['se']=float(score[i].std(ddof=1)/np.sqrt(3)) if rec['mean'] is not None else None
        candidates.append(rec)
    valid=np.isfinite(score).all(1)
    if not valid.any():raise RuntimeError(f'No valid models: {stem}')
    means=np.where(valid,np.nanmean(score,1),-np.inf);best=int(means.argmax())
    tolerance=float(score[best].std(ddof=1)/np.sqrt(3))
    eligible=[KS[i] for i in range(len(KS)) if valid[i] and means[i]>=means[best]-tolerance-1e-12]
    selected=min(eligible,key=lambda k:(abs(k-5),k))
    model=fit_model(x[final_idx],alg,selected,42,True)
    if not getattr(model,'converged_',True):raise RuntimeError(f'Final GMM did not converge: {stem}')
    labels=model.predict(x).reshape(shape).astype(np.int32)
    fold_best=[KS[int(np.nanargmax(np.where(valid,score[:,f],np.nan)))] for f in range(3)]
    stability=[float(adjusted_rand_score(prediction[selected,a],prediction[selected,b])) for a,b in [(0,1),(0,2),(1,2)]]
    meta={**common,'algorithm':alg,'selected_k':selected,'score_best_k':KS[best],
          'eligible_k':eligible,'one_se_tolerance':tolerance,'candidates':candidates,
          'fold_best_k':fold_best,'selection_at_upper_bound':selected==12,'score_best_at_upper_bound':KS[best]==12,
          'resampling_ari':stability,'mean_resampling_ari':float(np.mean(stability)),
          'fit_sample_count':len(final_idx),'score_name':'held-out silhouette' if alg=='kmeans' else 'held-out mean log-likelihood',
          'seconds':time.perf_counter()-start,'converged':bool(getattr(model,'converged_',True))}
    if alg=='gmm':
        probability=model.predict_proba(x).max(1).astype(np.float32)
        np.savez_compressed(d/f'{stem}_confidence.npz',confidence=probability.reshape(shape))
        meta.update(final_training_bic=float(model.bic(x[final_idx])),mean_max_posterior=float(probability.mean()))
    render_result(d,stem,labels,rgb,meta)

def hdbscan(x,shape,d,prefix,rgb,common):
    stem=f'{prefix}_hdbscan';completed=f'{prefix}_hdbscan_completed'
    if (d/f'{completed}.json').exists():return
    start=time.perf_counter();n=len(x);trials=[];models={};trees={}
    for ms in [3,5,10]:
        first=HDBSCAN(min_samples=ms,min_cluster_size=max(32,round(.005*n)),cluster_selection_method='eom',n_jobs=2,copy=False).fit(x)
        trees[ms]=first._single_linkage_tree_
        for fraction in [.005,.01,.02]:
            size=max(32,round(fraction*n))
            lab,prob=tree_to_labels(trees[ms].copy(),size,'eom',False,0.,None)
            model=SimpleNamespace(labels_=lab,probabilities_=prob)
            stats=summarize(model.labels_)
            rec={'min_samples':ms,'minimum_fraction':fraction,'min_cluster_size':size,'epsilon':0.,**stats}
            trials.append(rec);models[len(trials)-1]=model
    def key(i):
        r=trials[i];k=r['cluster_count'];noise=r['noise_fraction']
        # Feasible > count feasible > count violation; then maximum coverage.
        return (not(3<=k<=10 and noise<.15),not(3<=k<=10),max(3-k,0,k-10),noise,abs(k-5),i)
    idx=min(range(len(trials)),key=key)
    # Merge nearby groups only when every unmerged candidate exceeds ten.
    epsilon_errors=[]
    if all(t['cluster_count']>10 for t in trials):
        base=trials[idx]
        nn=NearestNeighbors(n_neighbors=base['min_samples'],n_jobs=2).fit(x)
        distances=nn.kneighbors(x[np.random.default_rng(9).choice(n,min(2048,n),False)])[0][:,-1]
        for factor in [1.,2.,4.,8.]:
            eps=float(np.median(distances)*factor)
            try:lab,prob=tree_to_labels(trees[base['min_samples']].copy(),base['min_cluster_size'],'eom',False,eps,None)
            except TypeError as error:
                epsilon_errors.append({'epsilon':eps,'error':str(error)});continue
            model=SimpleNamespace(labels_=lab,probabilities_=prob)
            trials.append({**{k:base[k] for k in ['min_samples','minimum_fraction','min_cluster_size']},'epsilon':eps,**summarize(model.labels_)})
            models[len(trials)-1]=model
        idx=min(range(len(trials)),key=key)
    model=models[idx];labels=model.labels_.astype(np.int32);strength=model.probabilities_.astype(np.float32)
    chosen=trials[idx];meta={**common,'algorithm':'hdbscan','trials':trials,'selected_trial':idx,'epsilon_errors':epsilon_errors,'parameters':{k:chosen[k] for k in ['min_samples','min_cluster_size','minimum_fraction','epsilon']},'target_met':3<=chosen['cluster_count']<=10 and chosen['noise_fraction']<.15,'seconds':time.perf_counter()-start}
    render_result(d,stem,labels.reshape(shape),rgb,meta)
    # Transparent optional completion, NOT native HDBSCAN soft probabilities.
    # Use feature-neighbour votes and a distance gate; never fill just to meet 85%.
    new=labels.copy();accepted=np.zeros(n,bool);agreement=np.zeros(n,np.float32)
    noise=np.flatnonzero(labels<0);cores=np.flatnonzero((labels>=0)&(strength>=.5));gates={}
    if len(noise) and len(cores)>=16:
        nn=NearestNeighbors(n_neighbors=16,n_jobs=2).fit(x[cores])
        calibration=np.random.default_rng(7).choice(len(cores),min(8192,len(cores)),False)
        cd,ci=nn.kneighbors(x[cores[calibration]])
        # Exclude self and calibrate the mean distance to 15 neighbours per group.
        means=cd[:,1:].mean(1)
        for k in np.unique(labels[cores]):
            use=labels[cores[calibration]]==k
            if use.sum()>=10:gates[int(k)]=float(2*np.quantile(means[use],.95))
        for begin in range(0,len(noise),4096):
            ii=noise[begin:begin+4096];dist,neighbors=nn.kneighbors(x[ii],n_neighbors=15)
            tags=labels[cores[neighbors]];ww=1/np.maximum(dist,1e-8)
            group=np.unique(labels[cores]);votes=np.stack([(ww*(tags==k)).sum(1) for k in group],1)
            best=votes.argmax(1);pred=group[best];confidence=votes.max(1)/votes.sum(1)
            gate=np.array([gates.get(int(k),0) for k in pred]);ok=(confidence>=.8)&(dist.mean(1)<=gate)
            new[ii[ok]]=pred[ok];accepted[ii[ok]]=True;agreement[ii]=confidence
    np.savez_compressed(d/f'{completed}_assignment.npz',newly_assigned=accepted.reshape(shape),neighbor_agreement=agreement.reshape(shape),original_strength=strength.reshape(shape))
    stats=summarize(new)
    extra={**common,'algorithm':'hdbscan_completed','parameters':meta['parameters'],'raw_noise_fraction':chosen['noise_fraction'],'newly_assigned_fraction':float(accepted.mean()),'distance_gates':gates,'target_met':3<=stats['cluster_count']<=10 and stats['noise_fraction']<.15,'assignment':'15 feature-space neighbours of HDBSCAN members with strength >=0.5; inverse-distance vote >=0.8; mean distance <=2x within-group 95th percentile core-neighbour distance. Votes are not calibrated probabilities. No XY.','seconds':time.perf_counter()-start}
    render_result(d,completed,new.reshape(shape),rgb,extra,base=labels.reshape(shape))

def analyze(task):
    image,method,dim=task;d=ROOT/image;f=d/f'{method}_features.npz'
    with threadpool_limits(limits=2):
        raw=np.load(f)['features'];shape=raw.shape[:2];x=np.ascontiguousarray(raw[:,:,:dim].reshape(-1,dim))
        assert np.isfinite(x).all();rgb=np.asarray(Image.open(d/'input.jpg'))
        common={'id':image,'method':method,'dimensions':dim,'feature_sha256':digest(f),'grid_shape':list(shape),'sample_count':len(x)}
        prefix=f'{method}_pc{dim}'
        for alg in ['kmeans','gmm']:automatic(x,shape,alg,d,f'{prefix}_{alg}',rgb,common)
        hdbscan(x,shape,d,prefix,rgb,common)
    print('DONE',image,method,dim,flush=True)
    return task

def protocol():
    return {'version':1,'script_sha256':digest(__file__),'images':50,'datasets':extraction.prior.DATASETS,'methods':METHODS,'dimensions':DIMS,
      'features':'Original frozen PCA transform; unwhitened first 3 or 16 scores; no RGB, XY, or additional L2 normalization. Local features: 384px enlargement, inverse-aligned four flips, ~75% overlap, Hann blending. Aspect-preserving longest-side256 sampled grids. Original calibration included six feature treatments.',
      'automatic_k':'K=3..12. Three spatial block folds (8x8 blocks, shuffled into 3 groups), 3072 train/1024 test samples per fold shared across K. KMeans held-out silhouette; full-covariance GMM held-out mean log-likelihood. Candidate must converge in all folds. Choose K closest to5 (lower on tie) among candidates within one SE of best mean score. SE is spatial resampling tolerance, not calibrated confidence; features retain overlapping/global context across block borders. Final fit8192 samples, predict all grid locations. Seeds fixed; final n_init10/3; GMM reg1e-5. BIC diagnostic only.',
      'hdbscan':'Fit all grid locations; min_samples3/5/10 x min_cluster_size.5/1/2% (floor32), EOM. Reuse one single-linkage tree per min_samples through sklearn1.9.1 tree_to_labels; independently verified against full refits. Prefer target3..10 and noise<15%; then count feasible, smaller count violation, lower noise, closer5. If all candidates>10, test epsilon median core-distance x1/2/4/8 on best configuration. No forced count or coverage. Separate guarded feature-neighbour completion map.',
      'distribution':'Empirical P(K=k)=number of images with k occupied groups / number of images in filter. Each feature approach and PC representation separate; 50 images overall,10 per dataset. Not a posterior, species count, or independent replication across feature treatments.',
      'limitations':'Original ten examples per dataset are a frozen illustrative collection, not a new representative random population sample. Feature groups are not species or individual objects. More coverage is not evidence of greater biological accuracy. All selections exploratory; no ground-truth masks used.'}

def run(workers):
    rows=extraction.prepare();save(ROOT/'protocol.json',protocol());tasks=[(r['id'],m,k) for r in rows for m in METHODS for k in DIMS]
    pending=set(tasks);active={};failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            finished=[t for t in pending if (ROOT/t[0]/f'{t[1]}_pc{t[2]}_hdbscan_completed.json').exists()]
            pending.difference_update(finished)
            for t in sorted(pending):
                if len(active)>=workers:break
                d=ROOT/t[0];f=d/f'{t[1]}_features.npz'
                # Metadata is written after the archive: readiness avoids partially written npz.
                if not f.exists() or not (d/f'{t[1]}_extraction.json').exists():continue
                active[pool.submit(analyze,t)]=t;pending.remove(t)
            if active:
                done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for future in done:
                    task=active.pop(future)
                    try:future.result()
                    except Exception as e:
                        import traceback;traceback.print_exc();failures.append({'task':task,'error':str(e)})
            elif pending:time.sleep(15)
            save(ROOT/'progress.json',{'total_cases':len(tasks),'completed_cases':sum((ROOT/t[0]/f'{t[1]}_pc{t[2]}_hdbscan_completed.json').exists() for t in tasks),'pending':len(pending),'active':list(active.values()),'failures':failures,'updated_unix':time.time()})
    if failures:raise RuntimeError(str(failures))
    save(ROOT/'analysis_complete.json',{'cases':400,'maps':1600})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();run(a.workers)
