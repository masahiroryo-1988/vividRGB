"""Frozen 50-image full-covariance GMM regularization and orthogonal varimax.

PCA basis columns, not image RGB or spatial coordinates, enter varimax. All
retained dimensions are rotated together; no whitening or factor-score scaling.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, hashlib, json, time, warnings, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import auto_clusters50 as prior
import pca_whitening10 as visual

BASE=prior.ROOT
ROOT=BASE.parent/'eco_gmm_final50_v1'
METHODS=prior.METHODS
DIMS=[3,16]
LAMBDAS=[1,2,4,8,16,32,64]
ALPHAS=[.001,.01,.1,1.,10.,100.]
KS=list(range(1,13))

def save(p,v): prior.save(p,v)
def read(p): return json.loads(Path(p).read_text())
def alpha_key(v): return f'a{v:g}'.replace('.','p')

def varimax(basis, seed=42, starts=5, max_iter=1000, tol=1e-9):
    """Orthogonal rotation of projection loadings; Kaiser row normalization.

    Optimize row-normalized PCA projection weights, then apply R to the original
    unnormalized basis and scores. This is basis rotation, not whitening or the
    alternative covariance-loading/factor-score convention.
    """
    b=np.asarray(basis,dtype=np.float64)
    h=np.linalg.norm(b,axis=1);a=b/np.maximum(h[:,None],1e-15)
    p,k=a.shape;best=None;rng=np.random.default_rng(seed)
    def objective(q):
        l=a@q
        return float(np.sum(l**4)-np.sum(np.sum(l*l,axis=0)**2)/p)
    for st in range(starts):
        r=np.eye(k) if st==0 else np.linalg.qr(rng.normal(size=(k,k)))[0]
        before=objective(r)
        for it in range(max_iter):
            l=a@r
            u,s,v=np.linalg.svd(a.T@(l**3-l@(np.diag(np.sum(l*l,axis=0))/p)),full_matrices=False)
            new=u@v;score=objective(new)
            if score+1e-8<before: raise RuntimeError('Varimax objective decreased')
            r=new
            if abs(score-before)<=tol*max(abs(before),1.):break
            before=score
        result={'rotation':r,'objective':objective(r),'iterations':it+1,'start':st,'converged':it+1<max_iter}
        if best is None or result['objective']>best['objective']:best=result
    # Deterministic factor ordering/sign: loading concentration then largest loading.
    r=best['rotation'];weights=b@r
    order=np.argsort(-np.sum((a@r)**4,axis=0),kind='stable');r=r[:,order];weights=weights[:,order]
    signs=np.where(weights[np.abs(weights).argmax(0),np.arange(k)]<0,-1.,1.);r=r*signs
    best.update(rotation=r,initial_objective=objective(np.eye(k)),kaiser_normalization=True,starts=starts)
    np.testing.assert_allclose(r.T@r,np.eye(k),atol=1e-12)
    return best

def fit_mle(train,k):
    with warnings.catch_warnings(record=True) as caught:
        model=GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-5,n_init=3,max_iter=300,tol=.001,random_state=42).fit(train)
        if not model.converged_:
            model.set_params(max_iter=600,warm_start=True,n_init=1);model.fit(train)
    return model,[str(w.message) for w in caught]

def rotate_model(model,r):
    """Exact coordinate transform of a fitted full-covariance mixture."""
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    import copy
    rotated=copy.deepcopy(model)
    rotated.means_=model.means_@r
    rotated.covariances_=np.einsum('ab,kac,cd->kbd',r,model.covariances_,r)
    rotated.precisions_cholesky_=_compute_precision_cholesky(rotated.covariances_,'full')
    rotated.precisions_=np.einsum('kij,kjl->kil',rotated.precisions_cholesky_,np.transpose(rotated.precisions_cholesky_,(0,2,1)))
    return rotated

def double_precision_model(model):
    """Evaluate the same fitted parameters with a consistent float64 factorization."""
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    import copy
    m=copy.deepcopy(model)
    m.means_=np.asarray(model.means_,np.float64)
    m.weights_=np.asarray(model.weights_,np.float64)
    c=np.asarray(model.covariances_,np.float64)
    # Float32 weighted covariance products can differ between the two triangles.
    # Restore symmetry before evaluating equivalent coordinate systems.
    m.covariances_=(c+np.transpose(c,(0,2,1)))/2
    m.precisions_cholesky_=_compute_precision_cholesky(m.covariances_,'full')
    m.precisions_=np.einsum('kij,kjl->kil',m.precisions_cholesky_,np.transpose(m.precisions_cholesky_,(0,2,1)))
    return m

def stats(labels, confidence=None):
    values,counts=np.unique(labels,return_counts=True);fractions=counts/labels.size
    out={'occupied_k':int(len(values)),'area_fractions':{str(int(v)):float(c) for v,c in zip(values,fractions)},'largest_group_fraction':float(fractions.max()),'groups_ge_1pct':int((fractions>=.01).sum()),'groups_ge_2pct':int((fractions>=.02).sum()),'small_group_area':float(fractions[fractions<.01].sum())}
    if confidence is not None:out['mean_max_responsibility']=float(np.mean(confidence))
    return out

def render(d,stem,labels,rgb,base,extra):
    labels=labels.astype(np.int32)
    np.savez_compressed(d/f'{stem}_labels.npz',labels=labels)
    visual.render(d,stem,labels,base,rgb)
    save(d/f'{stem}.json',{**extra,**stats(labels)})

def penalty_case(image,method,dim):
    d=ROOT/image;d.mkdir(exist_ok=True);src=BASE/image
    prefix=f'{method}_pc{dim}';out=d/f'{prefix}_penalty.json'
    if out.exists() and read(out).get('version')==2:return read(out)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        raw=np.load(src/f'{method}_features.npz')['features'];shape=raw.shape[:2]
        x=np.ascontiguousarray(raw[:,:,:dim].reshape(-1,dim),dtype=np.float64)
        _,ix=prior.partitions(shape);train_fit=np.ascontiguousarray(raw[:,:,:dim].reshape(-1,dim)[ix]);train=x[ix];n=len(train)
        tr=np.load(src/'transform.npz');vr=varimax(tr['basis'][:,:dim]);r=vr.pop('rotation')
        z=np.ascontiguousarray(x@r)
        reconstruction=float(np.max(np.abs((z@r.T)-x)))
        probe=np.random.default_rng(53).choice(len(x),min(2048,len(x)),False)
        distance_error=float(np.max(np.abs(np.linalg.norm(x[probe]-x[probe[::-1]],axis=1)-np.linalg.norm(z[probe]-z[probe[::-1]],axis=1))))
        np.savez_compressed(d/f'{prefix}_varimax.npz',rotation=r,basis=tr['basis'][:,:dim]@r)
        vr.update(reconstruction_max_error=reconstruction,pairwise_distance_max_error=distance_error,orthogonality_max_error=float(np.max(np.abs(r.T@r-np.eye(dim)))))
        rgb=np.asarray(Image.open(src/'input.jpg'));base=np.load(src/f'{prefix}_kmeans_labels.npz')['labels']
        # Consistent per-image display stretch for all four methods, calibrated from
        # equal fixed samples of the retained four fields (not one method at a time).
        pool=[]
        for m in METHODS:
            field=np.load(src/f'{m}_features.npz')['features'][:,:,:dim].reshape(-1,dim)
            choose=np.random.default_rng(31).choice(len(field),min(2048,len(field)),False)
            pool.append(field[choose]@r)
        low,high=np.quantile(np.concatenate(pool)[:,:3],[.02,.98],axis=0)
        color=np.uint8(np.clip((z[:,:3]-low)/np.maximum(high-low,1e-9),0,1)*255).reshape(*shape,3)
        Image.fromarray(color).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.BILINEAR).save(d/f'{prefix}_varimax.jpg',quality=95)
        # Refit the full set in float64 for internally consistent candidate
        # likelihoods. Float32 EM can change tiny amounts with BLAS thread count.
        previous=read(src/f'{prefix}_gmm_bic_min.json')
        assert previous['sample_index_sha256']==hashlib.sha256(ix.tobytes()).hexdigest()
        candidates=[]
        models={}
        for k in KS:
            model,ww=fit_mle(train,k);models[k]=model;p=k*(dim+dim*(dim+1)//2)+k-1
            ll=float(model.score(train)*n)
            np.testing.assert_allclose(model.bic(train),-2*ll+p*np.log(n),rtol=1e-10)
            candidates.append({'k':k,'log_likelihood':ll,'parameter_count':p,'converged':bool(model.converged_),'bic':float(model.bic(train)),'iterations':int(model.n_iter_),'warnings':ww})
        candidates.sort(key=lambda v:v['k']);valid=[v for v in candidates if v['converged']]
        selections=[]
        for lam in LAMBDAS:
            chosen=min(valid,key=lambda v:(-2*v['log_likelihood']+lam*v['parameter_count']*np.log(n),v['k']))
            scores=[{'k':v['k'],'score':float(-2*v['log_likelihood']+lam*v['parameter_count']*np.log(n))} for v in valid]
            selections.append({'lambda':lam,'selected_k':chosen['k'],'criterion':float(min(v['score'] for v in scores)),'scores':scores,'at_search_boundary':chosen['k'] in [1,12]})
        refit_differences=[];rotation_checks=[]
        for k in sorted({v['selected_k'] for v in selections}):
            model=models[k]
            if not model.converged_:raise RuntimeError(f'MLE did not converge {image} {method} {dim} {k}')
            old=next((v for v in previous['candidates'] if v['k']==k),None)
            if old is not None:refit_differences.append({'k':k,'log_likelihood_difference':float(model.score(train)*n-old['log_likelihood'])})
            model=double_precision_model(model);rot=rotate_model(model,r)
            np.testing.assert_allclose(model.score_samples(x[probe]),rot.score_samples(z[probe]),atol=1e-8)
            labels=rot.predict(z).reshape(shape)
            rotation_checks.append({'k':k,'max_log_density_error':float(np.max(np.abs(model.score_samples(x[probe])-rot.score_samples(z[probe])))),'prediction_ari':float(adjusted_rand_score(model.predict(x[probe]),rot.predict(z[probe])))})
            stem=f'{prefix}_gmm_k{k}';render(d,stem,labels,rgb,base,{'selected_k':k,'converged':bool(model.converged_),'rotation_applied':True,'fit_sample_count':n})
        # Orthogonality makes existing K-means selection/labels exactly reusable.
        km=read(src/f'{prefix}_kmeans.json')
        render(d,f'{prefix}_kmeans',base,rgb,base,{'selected_k':km['selected_k'],'mean_resampling_ari':km['mean_resampling_ari'],'orthogonal_invariance_reuse':True})
        result={'version':2,'id':image,'method':method,'dimensions':dim,'fit_sample_count':n,'feature_sha256':prior.digest(src/f'{method}_features.npz'),'candidates':candidates,'selections':selections,'varimax':vr,'rotation_checks':rotation_checks,'refit_differences':refit_differences,'kmeans':{'selected_k':km['selected_k'],'mean_resampling_ari':km['mean_resampling_ari']},'seconds':time.perf_counter()-start}
        save(out,result)
    print('PENALTY DONE',image,method,dim,round(result['seconds'],1),flush=True)
    return result

def bayes_case(image,method,dim,alpha,pilot=False):
    d=ROOT/image;d.mkdir(exist_ok=True);src=BASE/image;prefix=f'{method}_pc{dim}';key=alpha_key(alpha)
    out=d/f'{prefix}_bayes_{key}.json'
    if out.exists() and read(out)['converged']:return read(out)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        raw=np.load(src/f'{method}_features.npz')['features'];shape=raw.shape[:2]
        x=np.ascontiguousarray(raw[:,:,:dim].reshape(-1,dim),dtype=np.float64);_,ix=prior.partitions(shape)
        rr=d/f'{prefix}_varimax.npz'
        if rr.exists():r=np.load(rr)['rotation']
        else:r=varimax(np.load(src/'transform.npz')['basis'][:,:dim])['rotation']
        z=np.ascontiguousarray(x@r);train=z[ix]
        with warnings.catch_warnings(record=True) as caught:
            model=BayesianGaussianMixture(n_components=12,covariance_type='full',weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=alpha,reg_covar=1e-5,mean_precision_prior=1.,max_iter=500,tol=.001,n_init=1,random_state=42).fit(train)
            if not model.converged_:
                model.set_params(warm_start=True,max_iter=1000,n_init=1);model.fit(train)
            if not model.converged_:
                model.set_params(warm_start=True,max_iter=5000,n_init=1);model.fit(train)
        labels=model.predict(z).reshape(shape);probs=model.predict_proba(z)
        counts=stats(labels,probs.max(1));weights=model.weights_
        rgb=np.asarray(Image.open(src/'input.jpg'));base=np.load(src/f'{prefix}_kmeans_labels.npz')['labels']
        render(d,f'{prefix}_bayes_{key}',labels,rgb,base,{'alpha':alpha,'converged':bool(model.converged_),'rotation_applied':True,'fit_sample_count':len(ix)})
        result={'id':image,'method':method,'dimensions':dim,'alpha':alpha,'fit_sample_count':len(ix),'converged':bool(model.converged_),'iterations':int(model.n_iter_),'lower_bound':float(model.lower_bound_),'weights':weights.tolist(),'components_weight_ge_1pct':int((weights>=.01).sum()),'components_weight_ge_2pct':int((weights>=.02).sum()),'warnings':[str(w.message) for w in caught],'seconds':time.perf_counter()-start,**counts}
        save(out,result)
    print('BAYES DONE',image,method,dim,alpha,result['occupied_k'],result['converged'],round(result['seconds'],1),flush=True)
    return result

def prepare():
    ROOT.mkdir(exist_ok=True)
    selection=read(BASE/'selection.json');save(ROOT/'selection.json',selection)
    for row in selection['images']:
        d=ROOT/row['id'];d.mkdir(exist_ok=True)
        for f in ['input.jpg','source.json',*[f'{m}_pca.jpg' for m in METHODS]]:
            if not (d/f).exists():shutil.copy2(BASE/row['id']/f,d/f)
    return selection['images']

def run(mode,workers=5):
    rows=prepare();tasks=[]
    if mode=='penalty':tasks=[(r['id'],m,d) for r in rows for m in METHODS for d in DIMS];func=penalty_case
    elif mode=='bayes':tasks=[(r['id'],m,d,a) for r in rows for m in METHODS for d in DIMS for a in ALPHAS];func=bayes_case
    elif mode=='pilot':
        tasks=[(f'{ds}_01','s10',16,a,True) for ds in ['moin','fungal_network','neon','coralscapes','pmid'] for a in [.001,1.,100.]];func=bayes_case
    failures=[];completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active={pool.submit(func,*t):t for t in tasks}
        for f in as_completed(active):
            try:f.result();completed+=1
            except Exception as error:
                import traceback;traceback.print_exc();failures.append({'task':active[f],'error':str(error)})
            save(ROOT/f'{mode}_progress.json',{'total':len(tasks),'completed':completed,'failures':failures,'updated_unix':time.time()})
    if failures:raise RuntimeError(str(failures))
    save(ROOT/f'{mode}_complete.json',{'total':len(tasks),'completed':completed})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['penalty','bayes','pilot']);ap.add_argument('--workers',type=int,default=5)
    args=ap.parse_args();run(args.mode,args.workers)
