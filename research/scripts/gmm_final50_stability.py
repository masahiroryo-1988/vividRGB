"""Fixed-n blocked resampling on the frozen earlier ten-image subset.

Select all lambda settings independently in each fold; compare labels on common
probe coordinates. Shared DINO context means these folds are not independent.
"""
import argparse,time,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import gmm_final50 as g

def analyze(image,dim):
    d=g.ROOT/image;target=d/f's10_pc{dim}_stability.json'
    if target.exists():return
    raw=np.load(g.BASE/image/'s10_features.npz')['features'];shape=raw.shape[:2]
    x=np.ascontiguousarray(raw[:,:,:dim].reshape(-1,dim),dtype=np.float64)
    r=g.varimax(np.load(g.BASE/image/'transform.npz')['basis'][:,:dim])['rotation']
    yy,xx=np.indices(shape);h,w=shape
    blocks=(np.minimum(yy*8//h,7)*8+np.minimum(xx*8//w,7)).ravel()
    rng=np.random.default_rng(240103);order=rng.permutation(64);fold=np.empty(64,int);fold[order]=np.arange(64)%3
    probe=np.random.default_rng(53).choice(len(x),min(2048,len(x)),False)
    results=[];predictions={lam:[] for lam in g.LAMBDAS}
    with threadpool_limits(limits=1):
        for f in range(3):
            choices=np.flatnonzero(fold[blocks]!=f);idx=np.random.default_rng(70+f).choice(choices,min(8192,len(choices)),False)
            n=len(idx);models={};scores=[]
            for k in g.KS:
                model,ww=g.fit_mle(x[idx],k);models[k]=model
                scores.append({'k':k,'log_likelihood':float(model.score(x[idx])*n),'p':k*(dim+dim*(dim+1)//2)+k-1,'converged':bool(model.converged_)})
            choices=[]
            for lam in g.LAMBDAS:
                chosen=min([s for s in scores if s['converged']],key=lambda s:(-2*s['log_likelihood']+lam*s['p']*np.log(n),s['k']))
                k=chosen['k'];predictions[lam].append(models[k].predict(x[probe]))
                choices.append({'lambda':lam,'selected_k':k})
            results.append({'fold':f,'n':n,'selections':choices,'candidates':scores})
    summary=[]
    for lam in g.LAMBDAS:
        labs=predictions[lam]
        aris=[float(adjusted_rand_score(labs[a],labs[b])) for a,b in [(0,1),(0,2),(1,2)]]
        ks=[next(t['selected_k'] for t in v['selections'] if t['lambda']==lam) for v in results]
        summary.append({'lambda':lam,'fold_k':ks,'same_k_in_all_folds':len(set(ks))==1,'pairwise_ari':aris,'mean_ari':float(np.mean(aris))})
    g.save(target,{'id':image,'dimensions':dim,'method':'s10','folds':results,'summary':summary,'note':'Fixed8192 fit locations from two-thirds of8x8 spatial blocks per fold. Overlapping context remains. ARI is agreement, not biological accuracy.'})
    print('STABILITY DONE',image,dim,flush=True)

def run(workers):
    ids=[r['id'] for r in g.read(g.BASE.parent/'eco_clustering10_v1'/'selection.json')['images']]
    assert len(ids)==10
    tasks=[(i,d) for i in ids for d in g.DIMS]
    done=0;failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active={pool.submit(analyze,*t):t for t in tasks}
        for f in as_completed(active):
            try:f.result();done+=1
            except Exception as error:
                import traceback;traceback.print_exc();failures.append({'task':active[f],'error':str(error)})
            g.save(g.ROOT/'stability_progress.json',{'total':len(tasks),'completed':done,'failures':failures})
    if failures:raise RuntimeError(str(failures))
    g.save(g.ROOT/'stability_complete.json',{'cases':20,'fits':720,'images':ids})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();run(a.workers)
