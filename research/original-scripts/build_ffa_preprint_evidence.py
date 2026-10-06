"""Recompute manuscript summaries and figures from frozen results; no inference."""
import argparse, csv, hashlib, itertools, json, shutil
from pathlib import Path
import numpy as np
from scipy.stats import bootstrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from ffa_cost_tradeoff import build_cost_tradeoff

DOMAINS=['moin','fungal_network','neon','coralscapes','pmid']
NAMES=['Grassland','Fungal networks','Airborne forest','Coral reefs','Phytoplankton']
METHODS=['whole','sam3','s05','s10']
LABELS=['DINOv3','SAM3 automatic','FFA 5%','FFA 10%']
COLORS=['#556579','#9a6484','#087f8c','#dc8b2d']

def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def summary(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))
def metric(r,m): return r['methods'][m]['metrics']['quality'] if m=='sam3' else r['methods'][m]['k']['6']
def savefig(fig,out,name):
    fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
def run():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'figures').mkdir(exist_ok=True);(a.out/'data').mkdir(exist_ok=True)
    aliases={'ecology-collection75':'eco_collection75_v1','gmm-final50':'eco_gmm_final50_v1','phase1-benchmark':'moin_phase1_benchmark_v1','fourflip-benchmark':'moin_phase1_fourflip_v1','multiscale-pilot':'moin_multiscale_pilot_v1'}
    def folder(name):
        p=a.source/name
        return p if p.exists() else a.source/aliases[name]
    src=folder('ecology-collection75')/'results.json';raw=json.loads(src.read_text());rows=raw['images']
    assert len(rows)==50 and all(sum(x['dataset']==d for x in rows)==10 for d in DOMAINS)
    ev={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'domains':{},'contrasts':[],'cost':{},'clustering':{},'phase1':{},'selection':rows}
    points=[];tests=[];signs=np.asarray(list(itertools.product([-1,1],repeat=10)))
    for ds,name in zip(DOMAINS,NAMES):
        rs=[r for r in rows if r['dataset']==ds];scores={m:np.array([metric(r,m)['rgb_edge']['f1'] for r in rs]) for m in METHODS}
        ev['domains'][ds]={'label':name,'rgb':{m:summary(scores[m]) for m in METHODS}}
        if ds=='coralscapes':ev['domains'][ds]['annotated']={m:summary([metric(r,m)['reference']['boundary']['f1'] for r in rs]) for m in METHODS}
        for focus in ['s05','s10']:
            for base in ['whole','sam3']:
                delta=scores[focus]-scores[base];obs=abs(delta.mean());null=abs((signs*delta).mean(1))
                p=float((null>=obs-1e-14).mean());rng=np.random.default_rng(20261005)
                ci=np.quantile(delta[rng.integers(0,10,(20000,10))].mean(1),[.025,.975])
                tests.append({'dataset':ds,'focus':focus,'baseline':base,'mean_delta':float(delta.mean()),'ci95':ci.tolist(),'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'exact_two_sided_p':p})
        for r in rs:
            for m in METHODS:
                z=metric(r,m)
                points.append(dict(image=r['id'],dataset=ds,method=m,f1=z['rgb_edge']['f1'],precision=z['rgb_edge']['precision'],recall=z['rgb_edge']['recall'],boundary_density=z['boundary_density'],annotated_f1=z['reference']['boundary']['f1'] if z.get('reference') else '',input_sha256=r['input_sha256']))
    order=np.argsort([x['exact_two_sided_p'] for x in tests]);prev=0.
    for rank,i in enumerate(order):
        prev=max(prev,min(1.,(len(tests)-rank)*tests[i]['exact_two_sided_p']));tests[i]['holm_p']=prev
    ev['contrasts']=tests
    ev['statistics_protocol']={'unit':'image','n_per_domain':10,'contrasts':20,'test':'exact two-sided paired sign-flip randomization of mean difference, all 1024 assignments','multiplicity':'Holm over 20 domain x scale x baseline contrasts','ci':'20000 paired image bootstrap draws, percentile, seed20261005; descriptive and unadjusted','scope':'exploratory conditional analysis, not a preregistered generalization claim; image independence and sign exchangeability may fail within shared sites/acquisitions'}
    with (a.out/'data/per_image_scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    for m in ['s05','s10','sam3']:
        ex=[r['methods'][m]['extraction'] for r in rows]
        ev['cost'][m]={k:summary([z[k] for z in ex]) for k in ['seconds','gpu_board_joules','peak_gpu_mib']}
        ev['cost'][m]['inverse_joules']=summary([1/z['gpu_board_joules'] for z in ex])
        ev['cost'][m]['per_domain']={d:{k:summary([r['methods'][m]['extraction'][k] for r in rows if r['dataset']==d]) for k in ['seconds','gpu_board_joules']} for d in DOMAINS}
    ev['cost']['whole']={'seconds':summary([r['methods']['whole']['extraction']['seconds'] for r in rows]),'gpu_board_joules':None,'reason':'full extraction energy not measured continuously; cached encoding energy not interchangeable'}
    ev['cost']['paired_5_to_10']={k:summary([r['methods']['s05']['extraction'][k]/r['methods']['s10']['extraction'][k] for r in rows]) for k in ['seconds','gpu_board_joules']}
    g=json.loads((folder('gmm-final50')/'results.json').read_text());ev['clustering']['distributions']=g['distributions'];ev['clustering']['stability']=g['stability_summary'];ev['clustering']['software']=g['software'];ev['clustering']['protocol']=g['protocol']
    ev['clustering']['kmeans']={m:summary([r['results'][m]['16']['kmeans']['selected_k'] for r in g['images']]) for m in ['whole','s05','s10']}
    ev['clustering']['rotation_checks_example']=g['images'][0]['results']['s10']['16']['rotation_checks']
    for name in ['phase1-benchmark','fourflip-benchmark','multiscale-pilot']:
        p=folder(name)/'results.json'
        if p.exists():
            z=json.loads(p.read_text());ev['phase1'][name]={k:z[k] for k in ['completed','total','blocks','block_unit','summary','paired_contrasts','paired_deltas','protocol'] if k in z}
    for fname in ['dino_protocol.json','sam3_hf_protocol.json','validation.json']:
        shutil.copy2(folder('ecology-collection75')/fname,a.out/'data'/fname)
    shutil.copy2(folder('gmm-final50')/'protocol.json',a.out/'data/gmm_protocol.json')
    # Primary individual observations and paired mean differences.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,5,figsize=(12,3.9),sharey=True)
    for ax,ds,name in zip(axs,DOMAINS,NAMES):
        rs=[r for r in rows if r['dataset']==ds];v=np.array([[metric(r,m)['rgb_edge']['f1'] for m in METHODS] for r in rs])
        for y in v:ax.plot(range(4),y,color='#bec8cc',alpha=.45,lw=.6,zorder=0)
        for j,m in enumerate(METHODS):
            y=v[:,j];jitter=np.random.default_rng(42).uniform(-.085,.085,10)
            ax.scatter(j+jitter,y,c=COLORS[j],s=15,alpha=.75)
            ax.errorbar(j,y.mean(),yerr=y.std(ddof=1),fmt='D',c=COLORS[j],mec='white',ms=6,capsize=3,lw=1.5)
        ax.set_xticks(range(4),['DINOv3','SAM3','5%','10%'],rotation=35);ax.set_title(name);ax.set_ylim(0,.9);ax.grid(axis='y',alpha=.16)
    axs[0].set_ylabel('RGB-edge boundary F1');fig.tight_layout();savefig(fig,a.out,'f1_five_domains')
    plt.rcParams.update({'font.size':12})
    fig,ax=plt.subplots(figsize=(5.3,3));ds='coralscapes'
    for j,m in enumerate(METHODS):
        y=np.array([metric(r,m)['reference']['boundary']['f1'] for r in rows if r['dataset']==ds]);ax.scatter(j+np.linspace(-.06,.06,10),y,s=20,color=COLORS[j],alpha=.6);ax.errorbar(j,y.mean(),yerr=y.std(ddof=1),fmt='D',color=COLORS[j],ms=7,capsize=4)
    ax.set_xticks(range(4),LABELS);ax.set_ylabel('Annotated class-boundary F1');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);fig.tight_layout();savefig(fig,a.out,'coral_reference')
    # First frozen example per domain; avoid outcome-based figure selection.
    fig,axs=plt.subplots(5,5,figsize=(12,11.6));examples=[]
    for i,(ds,name) in enumerate(zip(DOMAINS,NAMES)):
        r=next(r for r in rows if r['dataset']==ds);examples.append(r['id']);d=folder('ecology-collection75')/r['id']
        paths=[d/'input.png',d/'whole_clusters.png',d/'sam3_hf_clusters.png',d/'s05_clusters.png',d/'s10_clusters.png']
        for j,p in enumerate(paths):
            im=Image.open(p).convert('RGB');im.thumbnail((800,800));axs[i,j].imshow(im);axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            for sp in axs[i,j].spines.values():sp.set_visible(False)
            if i==0:axs[i,j].set_title(['RGB','DINOv3 K=6','SAM3 automatic','FFA 5% K=6','FFA 10% K=6'][j],fontsize=14)
        axs[i,0].set_ylabel(name+'\n'+r['id'],fontsize=14)
    fig.subplots_adjust(wspace=.025,hspace=.08,left=.06,right=.995,top=.975,bottom=.015);savefig(fig,a.out,'qualitative_five_domains');ev['figure_examples']=examples
    # Independent display comparison on the same representative grassland image.
    fig,axs=plt.subplots(2,3,figsize=(9,6))
    for j,m in enumerate(['whole','s05','s10']):
        for i,f in enumerate([f'{m}_pca.jpg',f'{m}_pc16_varimax.jpg']):
            im=Image.open(folder('gmm-final50')/'moin_01'/f);axs[i,j].imshow(im);axs[i,j].axis('off')
        axs[0,j].set_title(['DINOv3','FFA 5%','FFA 10%'][j]);axs[0,j].text(-.05,.5,'PCA',transform=axs[0,j].transAxes,rotation=90,va='center');axs[1,j].text(-.05,.5,'PCA + varimax',transform=axs[1,j].transAxes,rotation=90,va='center')
    fig.tight_layout();savefig(fig,a.out,'pca_varimax')
    fig,axs=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
    for ax,m in zip(axs,['s05','s10']):
        dist=[x for x in g['distributions'] if x['dataset']=='all' and x['method']==m and x['dimensions']==16 and x['algorithm']=='penalty']
        ar=np.array([[z['probability_mass'].get(str(k),0) for k in range(1,13)] for z in dist])
        hm=ax.imshow(ar.T,origin='lower',aspect='auto',extent=[-.5,6.5,.5,12.5],vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(7),[str(z['setting']) for z in dist]);ax.set_yticks(range(1,13));ax.set_xlabel('Penalty multiplier '+r'$\lambda$');ax.set_title('FFA '+('5%' if m=='s05' else '10%'));ax.axvline(2,color='white',lw=1,ls='--')
    axs[0].set_ylabel('Selected mixture components K');fig.colorbar(hm,ax=axs,label='Fraction of 50 images',pad=.035);savefig(fig,a.out,'gmm_penalty')
    build_cost_tradeoff(rows,a.out,ev['source_sha256'])
    dump(a.out/'data/evidence.json',ev)
    print(json.dumps({'rgb':{d:ev['domains'][d]['rgb'] for d in DOMAINS},'coral':ev['domains']['coralscapes']['annotated'],'cost':ev['cost'],'tests':tests,'kmeans':ev['clustering']['kmeans']},indent=2))
if __name__=='__main__':run()
