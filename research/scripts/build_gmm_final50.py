"""Validate regularization experiment and build portable scientific inspection."""
import argparse,json,shutil,sys,zipfile,importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image
import gmm_final50 as g

def distribution(values):
    u,c=np.unique(values,return_counts=True)
    return {str(int(k)):float(v/len(values)) for k,v in zip(u,c)}

def build():
    root=g.ROOT;images=g.read(root/'selection.json')['images'];rows=[];manifest=set();label_count=0;cases=0;bayes=0;nonconverged=[];rotation=[];hash_checks=0
    for image in images:
        row=dict(image);row['results']={};d=root/row['id']
        for m in g.METHODS:
            row['results'][m]={}
            for dim in g.DIMS:
                stem=f'{m}_pc{dim}';q=g.read(d/f'{stem}_penalty.json');assert q['version']==2
                f=g.BASE/row['id']/f'{m}_features.npz';raw=np.load(f)['features'];sha=g.prior.digest(f);assert q['feature_sha256']==sha;hash_checks+=1
                chosen=[]
                for lam in g.LAMBDAS:
                    eligible=[c for c in q['candidates'] if c['converged']]
                    c=min(eligible,key=lambda v:(-2*v['log_likelihood']+lam*v['parameter_count']*np.log(q['fit_sample_count']),v['k']))
                    sel=next(s for s in q['selections'] if s['lambda']==lam);assert sel['selected_k']==c['k']
                    selected=sel['selected_k'];chosen.append(selected)
                    meta=g.read(d/f'{stem}_gmm_k{selected}.json');sel.update(occupied_k=meta['occupied_k'],largest_group_fraction=meta['largest_group_fraction'],small_group_area=meta['small_group_area'],groups_ge_1pct=meta['groups_ge_1pct'])
                assert all(a>=b for a,b in zip(chosen,chosen[1:])), 'Penalty must not increase selected count'
                r=np.load(d/f'{stem}_varimax.npz')['rotation'];np.testing.assert_allclose(r.T@r,np.eye(dim),atol=1e-12)
                assert q['varimax']['objective']>=q['varimax']['initial_objective']-1e-8
                assert q['varimax']['reconstruction_max_error']<1e-10 and q['varimax']['pairwise_distance_max_error']<1e-10
                rotation.extend(q['rotation_checks']);q['bayes']=[]
                required=[f'{stem}_kmeans',*[f'{stem}_gmm_k{k}' for k in set(chosen)]]
                for alpha in g.ALPHAS:
                    suffix=f'{stem}_bayes_{g.alpha_key(alpha)}';b=g.read(d/f'{suffix}.json')
                    # render JSON was saved before the detailed record and shares filename.
                    labels=np.load(d/f'{suffix}_labels.npz')['labels'];ss=g.stats(labels)
                    for key in ['occupied_k','area_fractions','largest_group_fraction','groups_ge_1pct','groups_ge_2pct','small_group_area']:assert ss[key]==b[key]
                    assert abs(sum(b['weights'])-1)<1e-10
                    assert b['components_weight_ge_1pct']==int((np.array(b['weights'])>=.01).sum())
                    if not b['converged']:nonconverged.append({'id':row['id'],'method':m,'dim':dim,'alpha':alpha})
                    q['bayes'].append(b);required.append(suffix);bayes+=1
                for name in set(required):
                    labels=np.load(d/f'{name}_labels.npz')['labels'];meta=g.read(d/f'{name}.json')
                    assert labels.shape==raw.shape[:2] and np.isfinite(labels).all() and labels.min()>=0
                    assert g.stats(labels)['occupied_k']==meta['occupied_k'];label_count+=1
                    for tail in ['_labels.npz','.json','_clusters.png','_overlay.jpg','_boundary.jpg']:
                        file=d/f'{name}{tail}';assert file.exists();manifest.add(file)
                for tail in ['_penalty.json','_varimax.npz','_varimax.jpg']:manifest.add(d/f'{stem}{tail}')
                q['stability']=g.read(d/f'{stem}_stability.json') if (d/f'{stem}_stability.json').exists() else None
                if q['stability']:manifest.add(d/f'{stem}_stability.json')
                row['results'][m][str(dim)]=q;cases+=1
        for name in ['input.jpg','source.json',*[f'{m}_pca.jpg' for m in g.METHODS]]:manifest.add(d/name)
        rows.append(row)
    assert cases==400 and bayes==2400 and len(rows)==50 and hash_checks==400
    assert not nonconverged, f'{len(nonconverged)} unconverged Bayesian models; resolve before concluding'
    assert all(v['prediction_ari']==1 for v in rotation)
    tests=g.read(root/'implementation_tests.json');assert tests['complete']
    stability_done=g.read(root/'stability_complete.json');assert stability_done['cases']==20
    datasets=['all',*g.prior.extraction.prior.DATASETS];distributions=[]
    for ds in datasets:
        cohort=[r for r in rows if ds=='all' or r['dataset']==ds]
        for m in g.METHODS:
            for dim in g.DIMS:
                qs=[r['results'][m][str(dim)] for r in cohort]
                for lam in g.LAMBDAS:
                    ss=[next(v for v in q['selections'] if v['lambda']==lam) for q in qs];kk=[v['selected_k'] for v in ss]
                    distributions.append({'dataset':ds,'method':m,'dimensions':dim,'algorithm':'penalty','setting':lam,'n':len(ss),'mean_k':float(np.mean(kk)),'median_k':float(np.median(kk)),'probability_mass':distribution(kk),'preferred_4_6_fraction':float(np.mean([(4<=k<=6) for k in kk])),'lower_1_2_fraction':float(np.mean([k<3 for k in kk])),'upper_12_fraction':float(np.mean([k==12 for k in kk])),'median_largest_group_fraction':float(np.median([v['largest_group_fraction'] for v in ss]))})
                for alpha in g.ALPHAS:
                    ss=[next(v for v in q['bayes'] if v['alpha']==alpha) for q in qs];kk=[v['occupied_k'] for v in ss]
                    distributions.append({'dataset':ds,'method':m,'dimensions':dim,'algorithm':'bayesian','setting':alpha,'n':len(ss),'mean_k':float(np.mean(kk)),'median_k':float(np.median(kk)),'probability_mass':distribution(kk),'preferred_4_6_fraction':float(np.mean([4<=k<=6 for k in kk])),'mean_components_weight_ge_1pct':float(np.mean([v['components_weight_ge_1pct'] for v in ss])),'median_largest_group_fraction':float(np.median([v['largest_group_fraction'] for v in ss])),'converged_fraction':float(np.mean([v['converged'] for v in ss]))})
    stability=[]
    for dim in g.DIMS:
        ss=[r['results']['s10'][str(dim)]['stability'] for r in rows if r['results']['s10'][str(dim)]['stability']]
        assert len(ss)==10
        for lam in g.LAMBDAS:
            records=[next(v for v in s['summary'] if v['lambda']==lam) for s in ss]
            ks=np.array([v['fold_k'] for v in records]);ari=[v['mean_ari'] for v in records]
            stability.append({'dimensions':dim,'lambda':lam,'images':10,'resampled_fits':30,'mean_k':float(ks.mean()),'mean_ari':float(np.mean(ari)),'median_ari':float(np.median(ari)),'same_k_fraction':float(np.mean([v['same_k_in_all_folds'] for v in records])),'min_ari':float(min(ari))})
    validation={'complete':True,'images':50,'cases':cases,'mle_candidate_fits':cases*12,'lambda_selections':cases*7,'bayesian_models':bayes,'bayesian_converged':bayes-len(nonconverged),'label_grids_checked':label_count,'rotation_model_checks':len(rotation),'maximum_log_density_error':float(max(v['max_log_density_error'] for v in rotation)),'minimum_prediction_ari':float(min(v['prediction_ari'] for v in rotation)),'retained_basis_checks':cases,'varimax_converged':sum(r['results'][m][str(dim)]['varimax']['converged'] for r in rows for m in g.METHODS for dim in g.DIMS),'stability_cases':20,'stability_candidate_fits':720}
    assert validation['varimax_converged']==400
    protocol=g.read(Path(__file__).with_name('gmm_final50_protocol.json'))
    protocol['mle_gmm']='Full covariance, n_init3,seed42,reg_covar1e-5,tol0.001,max_iter300,retry600. Refit ALL K1..12 in float64 on the same fixed8192-or-fewer coordinates; sample hash checked against prior experiment. Float64 ensures internally consistent candidate fits; tiny numerical/local-optimum differences from earlier float32 fits are reported. Score=-2*sum(logdensity)+lambda*p*log(n), p=K*(d+d*(d+1)/2)+K-1. Minimum converged score; smallerK exact ties. Lambda1 ordinaryBIC; higher values explicitly stronger penalty. No scores compared across dimensionalities.'
    protocol['mle_maps']='Use each selected fitted full-covariance model, transform means as means R and covariances as R.T covariance R, and verify pointwise density and predicted partition invariance. Maximum-responsibility assignment across all grid locations; no new DINO extraction.'
    protocol['bayesian_gmm']+=' For cases exhausting the initial budget, extend with a5000-iteration warm continuation at the same tolerance; all2400 final models must converge. Warnings from earlier stages are retained alongside the final convergence flag.'
    protocol['stability']='Earlier frozen ten-image subset, two per domain, local10% both3/16 dimensions. Three fits from two-thirds of8x8 spatial blocks per image; each uses8192 locations, same n as primary. Refit/select all K1..12 for every lambda in each fold; compare selected maps on fixed2048 common probe locations viaARI. Count variation andARI describe repeatability; overlapping/global DINO context means spatial folds remain dependent. No ground truth.'
    software={'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','Pillow']},'feature_source':str(g.BASE)}
    data={'images':rows,'distributions':distributions,'stability_summary':stability,'validation':validation,'protocol':protocol,'implementation_tests':tests,'software':software}
    for name,item in [('results.json',data),('distributions.json',distributions),('validation.json',validation),('protocol.json',protocol),('stability_summary.json',stability),('software.json',software)]:g.save(root/name,item);manifest.add(root/name)
    template=Path(__file__).with_name('gmm_final50_template.html').read_text(encoding='utf-8')
    (root/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')),encoding='utf-8');manifest.add(root/'index.html')
    primary=[v for v in distributions if v['dataset']=='all' and v['method']=='s10']
    summary={'validation':validation,'local10_all50':primary,'stability':stability,'alpha_count_changes':sum(len(set(v['occupied_k'] for v in r['results'][m][str(dim)]['bayes']))>1 for r in rows for m in g.METHODS for dim in g.DIMS)}
    g.save(root/'final_summary.json',summary);manifest.add(root/'final_summary.json');manifest.add(root/'implementation_tests.json')
    for name in ['gmm_final50.py','gmm_final50_stability.py','test_gmm_final50.py','build_gmm_final50.py','gmm_final50_template.html','gmm_final50_protocol.json']:
        shutil.copy2(Path(__file__).with_name(name),root/name);manifest.add(root/name)
    figures(data,manifest)
    # Keep displayed maps at512px longest side: 256-cell labels still fully resolved.
    for p in manifest:
        if p.parent!=root and p.suffix in ['.jpg','.png']:
            with Image.open(p) as image:
                if max(image.size)<=512:continue
                ratio=512/max(image.size);size=(round(image.width*ratio),round(image.height*ratio))
                display=image.resize(size,Image.Resampling.NEAREST if p.name.endswith('_clusters.png') else Image.Resampling.LANCZOS)
                if p.suffix=='.jpg':display.save(p,quality=95)
                else:display.save(p)
    with zipfile.ZipFile(root/'label_grids.zip','w',zipfile.ZIP_DEFLATED) as label_archive:
        for p in sorted(manifest):
            if p.name.endswith('_labels.npz') or p.name.endswith('_varimax.npz'):label_archive.write(p,p.relative_to(root))
        for name in ['protocol.json','selection.json','validation.json']:label_archive.write(root/name,name)
    manifest.add(root/'label_grids.zip');manifest.add(root/'selection.json')
    g.save(root/'asset_manifest.json',{'files':[str(p.relative_to(root)) for p in sorted(manifest)],'scientific_maps':'512px preview resolves longest-side256 label grids; original feature grids remain in server source directory.'});manifest.add(root/'asset_manifest.json')
    with zipfile.ZipFile(root.parent/'eco_gmm_final50_report.zip','w',zipfile.ZIP_DEFLATED) as report_archive:
        for p in sorted(manifest):
            if p.name.endswith('_labels.npz') or p.name.endswith('_varimax.npz'):continue
            report_archive.write(p,p.relative_to(root))
    g.save(root/'report_ready.json',validation);print(json.dumps(summary))

def figures(data,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cohorts=[('all',m,d) for m in g.METHODS for d in g.DIMS]+[(ds,'s10',16) for ds in g.prior.extraction.prior.DATASETS]
    for ds,m,dim in cohorts:
        rows=[r for r in data['images'] if ds=='all' or r['dataset']==ds]
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for column,(alg,settings,title) in enumerate([('penalty',g.LAMBDAS,'Strengthened penalty λ'),('bayesian',g.ALPHAS,'Bayesian concentration α')]):
            records=[v for v in data['distributions'] if v['dataset']==ds and v['method']==m and v['dimensions']==dim and v['algorithm']==alg]
            grid=np.array([[next((v['probability_mass'].get(str(k),0) for v in records if v['setting']==s),0) for k in range(1,13)] for s in settings])
            ax=axes[0,column];im=ax.imshow(grid,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            ax.set(xticks=np.arange(12),xticklabels=np.arange(1,13),yticks=np.arange(len(settings)),yticklabels=[f'{s:g}' for s in settings],xlabel='Selected K' if alg=='penalty' else 'Occupied groups',ylabel=title,title=f'{title}: empirical distribution, n={len(rows)}')
            for yy in range(len(settings)):
                for xx in range(12):
                    if grid[yy,xx]>0:ax.text(xx,yy,f'{100*grid[yy,xx]:.0f}',ha='center',va='center',fontsize=7,color='white' if grid[yy,xx]>.5 else '#253535')
            fig.colorbar(im,ax=ax,label='Empirical probability')
            ax=axes[1,column];matrix=[]
            for row in rows:
                q=row['results'][m][str(dim)]
                matrix.append([next(v['selected_k'] if alg=='penalty' else v['occupied_k'] for v in q['selections' if alg=='penalty' else 'bayes'] if v['lambda' if alg=='penalty' else 'alpha']==s) for s in settings])
            matrix=np.array(matrix)
            for i,values in enumerate(matrix):
                jitter=((i*17)%31-15)/100
                ax.plot(np.arange(len(settings))+jitter,values,color='#91a1a0',alpha=.25,lw=.6,marker='o',ms=2)
            ax.plot(range(len(settings)),matrix.mean(0),color='#137568',lw=2.5,marker='o',label='Mean across images')
            ax.axhspan(3.5,6.5,color='#e5efd9',alpha=.6)
            ax.set(xticks=np.arange(len(settings)),xticklabels=[f'{s:g}' for s in settings],yticks=range(1,13),ylim=(.5,12.5),xlabel=title,ylabel='Selected / occupied K',title='Paired images and mean; shaded preference 4–6')
            ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{ds} · {m} · {dim} retained PCA dimensions, with orthogonal varimax\nCounts are feature groups; no species or object-instance accuracy claim',fontsize=12)
        for ext in ['svg','png','pdf']:
            p=g.ROOT/f'regularization_{ds}_{m}_pc{dim}.{ext}';fig.savefig(p,dpi=180);manifest.add(p)
        plt.close(fig)

if __name__=='__main__':build()
