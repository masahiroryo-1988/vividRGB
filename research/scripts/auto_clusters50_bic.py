"""GMM BIC sensitivity on the same fixed final-fit sample as the CV analysis."""
import time,json,warnings
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
import auto_clusters50 as a

def analyze(task):
    image,method,dim=task;d=a.ROOT/image;prefix=f'{method}_pc{dim}'
    with threadpool_limits(limits=2):
        f=d/f'{method}_features.npz';grid=np.load(f)['features'];shape=grid.shape[:2]
        x=np.ascontiguousarray(grid[:,:,:dim].reshape(-1,dim));_,index=a.partitions(shape);train=x[index]
        candidates=[];models={};start=time.perf_counter()
        for k in a.KS:
            with warnings.catch_warnings(record=True) as caught:model=a.fit_model(train,'gmm',k,42,True)
            p=k*(dim+dim*(dim+1)//2)+k-1
            log_likelihood=float(model.score(train)*len(train))
            bic=float(model.bic(train));np.testing.assert_allclose(bic,-2*log_likelihood+p*np.log(len(train)),rtol=1e-10)
            candidates.append({'k':k,'bic':bic,'log_likelihood':log_likelihood,'parameter_count':p,'converged':bool(model.converged_),'iterations':int(model.n_iter_),'warnings':[str(w.message) for w in caught]})
            models[k]=model
        valid=[r for r in candidates if r['converged']]
        if not valid:raise RuntimeError(f'No converged BIC models: {task}')
        best=min(valid,key=lambda r:r['bic']);minimum=best['bic']
        for r in candidates:r['delta_bic']=r['bic']-minimum
        common={'id':image,'method':method,'dimensions':dim,'feature_sha256':a.digest(f),'grid_shape':list(shape),'sample_count':len(x),'fit_sample_count':len(index),'candidates':candidates,'score_best_k':best['k'],'score_best_at_upper_bound':best['k']==12,'minimum_bic':minimum,'score_name':'BIC (lower is better)','seconds':time.perf_counter()-start,'sample_index_sha256':__import__('hashlib').sha256(index.tobytes()).hexdigest()}
        rgb=np.asarray(Image.open(d/'input.jpg'))
        for alg,tolerance in [('gmm_bic_min',0.),('gmm_bic',6.)]:
            eligible=[r['k'] for r in valid if r['bic']<=minimum+tolerance+1e-10]
            selected=min(eligible);model=models[selected];labels=model.predict(x).reshape(shape).astype(np.int32)
            probability=model.predict_proba(x).max(1).astype(np.float32)
            meta={**common,'algorithm':alg,'selected_k':selected,'eligible_k':eligible,'bic_tolerance':tolerance,'selected_delta_bic':next(r['delta_bic'] for r in candidates if r['k']==selected),'selection_at_upper_bound':selected==12,'converged':True,'mean_max_posterior':float(probability.mean()),'selection_rule':'Smallest converged K within 6 BIC units of minimum; six is a prespecified practical tolerance, not a calibrated confidence interval.' if tolerance else 'Minimum BIC; smaller K only for exact ties.'}
            stem=f'{prefix}_{alg}';np.savez_compressed(d/f'{stem}_confidence.npz',confidence=probability.reshape(shape))
            a.render_result(d,stem,labels,rgb,meta)
    print('BIC DONE',*task,flush=True)

if __name__=='__main__':
    rows=json.loads((a.ROOT/'selection.json').read_text())['images'];tasks=[(r['id'],m,k) for r in rows for m in a.METHODS for k in a.DIMS]
    pending=set(tasks);active={};failures=[]
    a.save(a.ROOT/'bic_protocol.json',{'script_sha256':a.digest(__file__),'candidate_k':a.KS,'covariance_type':'full','sample':'Same fixed seed8192-or-fewer uniform grid locations used for CV-selected final GMM; all K fit and scored on identical sample. Scores not compared across PC representations. Correlated interpolation/context remains; treat BIC as exploratory model comparison, not proof of true component count.','bic':'-2*sum(log density) + p*log(n); p=K*(d+d*(d+1)/2)+(K-1). Verified numerically against sklearn BIC.','variants':{'gmm_bic_min':'Minimum BIC','gmm_bic':'Smallest K with BIC <= minimum + 6. Prespecified practical preference, not a confidence interval or guarantee of K4–6.'},'max_iter':300,'retry_max_iter':600,'n_init':3,'reg_covar':1e-5,'seed':42})
    with ProcessPoolExecutor(max_workers=2) as pool:
        while pending or active:
            pending.difference_update([t for t in pending if (a.ROOT/t[0]/f'{t[1]}_pc{t[2]}_gmm_bic.json').exists()])
            for task in sorted(pending):
                if len(active)>=2:break
                d=a.ROOT/task[0]
                if not (d/f'{task[1]}_extraction.json').exists():continue
                active[pool.submit(analyze,task)]=task;pending.remove(task)
            if active:
                done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:
                    task=active.pop(f)
                    try:f.result()
                    except Exception as error:
                        import traceback;traceback.print_exc();failures.append({'task':task,'error':str(error)})
            elif pending:time.sleep(15)
            a.save(a.ROOT/'bic_progress.json',{'total_cases':400,'completed_cases':sum((a.ROOT/t[0]/f'{t[1]}_pc{t[2]}_gmm_bic.json').exists() for t in tasks),'active':list(active.values()),'failures':failures,'updated_unix':time.time()})
    if failures:raise RuntimeError(str(failures))
    a.save(a.ROOT/'bic_complete.json',{'cases':400,'maps':800})
