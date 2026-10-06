"""Validate all maps, assemble a portable report, and export empirical distributions."""
import argparse,json,shutil,zipfile,sys,importlib.metadata
from pathlib import Path
import numpy as np
import auto_clusters50 as run
ALGORITHMS=['kmeans','gmm','gmm_bic_min','gmm_bic','hdbscan','hdbscan_completed']

def build(require_complete=False):
    root=run.ROOT;images=json.loads((root/'selection.json').read_text())['images'];maps=[];feature_checks=0;label_checks=0;gmm_converged=0
    for row in images:
        row['results']={};row['available_features']=[];d=root/row['id']
        for method in run.METHODS:
            row['results'][method]={}
            f=d/f'{method}_features.npz'
            if not f.exists() or not (d/f'{method}_extraction.json').exists():continue
            grid=np.load(f)['features'];sha=run.digest(f)
            row['available_features'].append(method)
            assert sha==json.loads((d/f'{method}_extraction.json').read_text())['feature_sha256'];feature_checks+=1
            for dim in run.DIMS:
                row['results'][method][str(dim)]={}
                for alg in ALGORITHMS:
                    stem=f'{method}_pc{dim}_{alg}';p=d/f'{stem}.json'
                    if not p.exists():continue
                    meta=json.loads(p.read_text());lab=np.load(d/f'{stem}_labels.npz')['labels']
                    if 'score_best_k' in meta:meta['score_best_at_lower_bound']=meta['score_best_k']==3
                    assert sha==meta['feature_sha256'] and lab.shape==grid.shape[:2]
                    stats=run.summarize(lab)
                    for key in ['cluster_count','noise_fraction','area_fractions']:assert stats[key]==meta[key]
                    if alg in ['kmeans','gmm']:
                        assert meta['noise_fraction']==0 and meta['selected_k'] in range(3,13)
                        valid=[c for c in meta['candidates'] if c['mean'] is not None]
                        best=max(valid,key=lambda c:c['mean']);assert best['k']==meta['score_best_k']
                        eligible=[c['k'] for c in valid if c['mean']>=best['mean']-meta['one_se_tolerance']-1e-12]
                        assert eligible==meta['eligible_k'] and min(eligible,key=lambda k:(abs(k-5),k))==meta['selected_k']
                    if alg.startswith('gmm_bic'):
                        valid=[c for c in meta['candidates'] if c['converged']]
                        best=min(valid,key=lambda c:c['bic']);assert best['k']==meta['score_best_k']
                        eligible=[c['k'] for c in valid if c['bic']<=best['bic']+meta['bic_tolerance']+1e-10]
                        assert eligible==meta['eligible_k'] and min(eligible)==meta['selected_k']
                        assert meta['noise_fraction']==0
                    if alg.startswith('gmm'):assert meta['converged'];gmm_converged+=1
                    if alg=='hdbscan_completed':
                        raw=np.load(d/f'{method}_pc{dim}_hdbscan_labels.npz')['labels'];named=raw>=0
                        np.testing.assert_array_equal(lab[named],raw[named])
                        assert set(np.unique(lab[lab>=0]))==set(np.unique(raw[raw>=0]))
                        assert meta['noise_fraction']<=float((raw<0).mean())
                        assert np.isclose(meta['newly_assigned_fraction'],np.mean((raw<0)&(lab>=0)))
                    for kind,ext in [('clusters','png'),('overlay','jpg'),('boundary','jpg')]:assert (d/f'{stem}_{kind}.{ext}').exists()
                    row['results'][method][str(dim)][alg]=meta;maps.append(meta);label_checks+=1
    validation={'complete':len(maps)==2400,'images':len(images),'maps':len(maps),'feature_hash_checks':feature_checks,'label_checks':label_checks,'gmm_converged':gmm_converged,'expected_maps':2400,'epsilon_errors':sum(len(r.get('epsilon_errors',[])) for r in maps)}
    if require_complete:assert validation['complete'] and feature_checks==200 and gmm_converged==1200
    distributions=[]
    for dataset in ['all',*run.extraction.prior.DATASETS]:
        ids={r['id'] for r in images if dataset=='all' or r['dataset']==dataset}
        for method in run.METHODS:
            for dim in run.DIMS:
                for alg in ALGORITHMS:
                    mm=[m for m in maps if m['id'] in ids and m['method']==method and m['dimensions']==dim and m['algorithm']==alg]
                    if not mm:continue
                    k,count=np.unique([m['cluster_count'] for m in mm],return_counts=True)
                    dist={str(int(kk)):float(cc/len(mm)) for kk,cc in zip(k,count)};assert abs(sum(dist.values())-1)<1e-12
                    distributions.append({'dataset':dataset,'method':method,'dimensions':dim,'algorithm':alg,'n':len(mm),'probability_mass':dist,'mean_k':float(np.mean([m['cluster_count'] for m in mm])),'preferred_4_6_fraction':float(np.mean([4<=m['cluster_count']<=6 for m in mm])),'mean_coverage':float(np.mean([1-m['noise_fraction'] for m in mm])),'target_met_fraction':float(np.mean([3<=m['cluster_count']<=(10 if alg.startswith('hdb') else 12) and m['noise_fraction']<.15 for m in mm]))})
    data={'images':images,'validation':validation,'distributions':distributions,'protocol':json.loads((root/'protocol.json').read_text()),'bic_protocol':json.loads((root/'bic_protocol.json').read_text())}
    software={'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','torch','transformers','Pillow']},'model':run.extraction.prior.p1.MODEL,'revision':run.extraction.prior.p1.REV}
    run.save(root/'software.json',software);data['software']=software
    run.save(root/'results.json',data);run.save(root/'validation.json',validation);run.save(root/'distributions.json',distributions)
    template=Path(__file__).with_name('auto_clusters50_template.html').read_text(encoding='utf-8')
    (root/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    for name in ['auto_clusters50.py','auto_clusters50_bic.py','auto_clusters50_extract.py','build_auto_clusters50.py','auto_clusters50_template.html','test_auto_clusters50.py','audit_auto_clusters50_features.py']:shutil.copy2(Path(__file__).with_name(name),root/name)
    if validation['complete']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors=['#187a68','#4679a2','#b77723','#a45450']
        names=['K-means','GMM · BIC, prefer smaller K','HDBSCAN','HDBSCAN + guarded assignment']
        for method in run.METHODS:
            for dim in run.DIMS:
                fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
                for ax,alg,name,color in zip(axes.flat,['kmeans','gmm_bic','hdbscan','hdbscan_completed'],names,colors):
                    item=next(r for r in distributions if r['dataset']=='all' and r['method']==method and r['dimensions']==dim and r['algorithm']==alg)
                    p=np.zeros(14)
                    for k,value in item['probability_mass'].items():p[min(int(k),13)]+=value
                    ax.axvspan(3.5,6.5,color='#edf4e7',zorder=0)
                    ax.bar(range(14),p,color=color,width=.75,zorder=2)
                    ax.set(title=f'{name} | mean K={item["mean_k"]:.1f}',ylim=(0,1),xticks=range(14),xticklabels=[*map(str,range(13)),'13+'],xlabel='Occupied feature groups (K)',ylabel='Empirical probability')
                    ax.grid(axis='y',alpha=.18,zorder=0);ax.spines[['top','right']].set_visible(False)
                method_name={'whole':'DINOv3 whole','s05':'Local 5%','s10':'Local 10%','s20':'Local 20%'}[method]
                fig.suptitle(f'{method_name} | first {dim} PCs | 50 images (10 per domain)\nShading: preferred 4–6 groups; counts do not represent species',fontsize=12)
                for ext in ['svg','png']:fig.savefig(root/f'distribution_{method}_pc{dim}.{ext}',dpi=180)
                plt.close(fig)
        with zipfile.ZipFile(root/'label_grids.zip','w',zipfile.ZIP_DEFLATED) as z:
            for row in images:
                for f in (root/row['id']).glob('*_labels.npz'):z.write(f,f.relative_to(root))
            for name in ['protocol.json','selection.json','validation.json','distributions.json']:z.write(root/name,name)
        with zipfile.ZipFile(root.parent/'eco_auto_clusters50_report.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in root.rglob('*'):
                if f.is_file() and not f.name.endswith('_original_pca.png') and (f.suffix in ['.html','.json','.py','.jpg','.png','.svg'] or f.name=='label_grids.zip'):z.write(f,f.relative_to(root))
    print(json.dumps(validation))
    return data

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--complete',action='store_true');args=ap.parse_args();build(args.complete)
