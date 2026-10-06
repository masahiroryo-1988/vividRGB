"""Aggregate the frozen 20-photo pilot and build figures and a portable report."""
import argparse
import csv
import html
import json
import zipfile
from pathlib import Path

import numpy as np

ORDER=['s05','s10','s20','s40','s05_10','s10_20','s05_10_20','s10_20_40','s05_10_20_40']
LABELS={m:' + '.join(str(int(s))+'%' for s in m[1:].split('_')) for m in ORDER}
SITES={'fessbach':'Feßbach','heidesheim':'Heidesheim','rettmer':'Rettmer','ribbeck':'Ribbeck'}
COLORS={'fessbach':'#24826a','heidesheim':'#cb743b','rettmer':'#6b65b3','ribbeck':'#3186a6'}
METRICS=['f1','precision','recall','seam_distance','boundary_density','effective_clusters','encoding_cost_ratio']
ADDITIONS=[('s05','s05_10'),('s10','s05_10'),('s10','s10_20'),
           ('s05_10','s05_10_20'),('s10_20','s10_20_40'),('s05_10_20','s05_10_20_40')]


def val(r,m,metric,k=6):
    q=r['methods'][m]; kk=q['k'][str(k)]
    if metric in ['f1','precision','recall']: return kk['rgb_edge'][metric]
    if metric=='seam_distance': return abs(q['seam_grids']['10']['ratio']-1)
    if metric=='encoding_cost_ratio': return q['estimated_encoding_seconds']/r['methods']['s10']['estimated_encoding_seconds']
    return kk[metric]


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def aggregate(out):
    selection=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    rows=[json.loads((out/r['id']/'metrics.json').read_text(encoding='utf-8')) for r in selection['images']]
    assert len(rows)==20 and len({r['source_sha256'] for r in rows})==20
    assert all(set(r['methods'])==set(ORDER) for r in rows)
    assert all(r['checks']['baseline_reproduction_passed'] for r in rows)
    assert len({r['protocol_hash'] for r in rows})==1
    sites=sorted(SITES); rng=np.random.default_rng(20261002)
    indices={site:np.array([i for i,r in enumerate(rows) if r['site']==site]) for site in sites}
    blocks={s:sorted({rows[i]['session'] for i in indices[s]}) for s in sites}
    # A paired cluster bootstrap within each site; each replicate keeps equal site weight.
    weights=np.zeros((5000,len(rows)),np.float64)
    for b in range(len(weights)):
        for site in sites:
            ids=[]
            for session in rng.choice(blocks[site],len(blocks[site]),replace=True):
                ids.extend(i for i in indices[site] if rows[i]['session']==session)
            for i in ids: weights[b,i]+=1/(4*len(ids))
    np.testing.assert_allclose(weights.sum(1),1)
    def summarize(v):
        boot=weights@v
        return {'mean':float(v.mean()),'median':float(np.median(v)),
                'ci95':np.quantile(boot,[.025,.975]).tolist(),'n':len(v),
                'site_means':{s:float(v[ix].mean()) for s,ix in indices.items()}}
    summary={}; deltas={}
    for metric in METRICS:
        summary[metric]={}; deltas[metric]={}
        baseline=np.array([val(r,'s10',metric) for r in rows])
        for m in ORDER:
            v=np.array([val(r,m,metric) for r in rows])
            summary[metric][m]=summarize(v)
            d=v-baseline
            deltas[metric][m]={**summarize(d),'positive':int((d>0).sum()),'negative':int((d<0).sum()),'ties':int((d==0).sum())}
    sensitivity={}
    for k in (4,6,8):
        sensitivity[str(k)]={}
        base=np.array([val(r,'s10','f1',k) for r in rows])
        for m in ORDER:
            v=np.array([val(r,m,'f1',k) for r in rows])
            sensitivity[str(k)][m]={'score':summarize(v),'delta':summarize(v-base)}
    seam_sensitivity={}
    for grid in [5,10,20,40]:
        seam_sensitivity[str(grid)]={}
        for m in ORDER:
            v=np.array([abs(r['methods'][m]['seam_grids'][str(grid)]['ratio']-1) for r in rows])
            base=np.array([abs(r['methods']['s10']['seam_grids'][str(grid)]['ratio']-1) for r in rows])
            seam_sensitivity[str(grid)][m]={'score':summarize(v),'delta':summarize(v-base)}
    additions=[]
    for a,b in ADDITIONS:
        delta=np.array([val(r,b,'f1')-val(r,a,'f1') for r in rows])
        additions.append({'from':a,'to':b,**summarize(delta),'positive':int((delta>0).sum()),
                          'definition':'Refit separate clusters after equal-weight feature fusion; mixtures are reweighted equally when a scale is added.'})
    data={'version':'multiscale-pilot-v1.0','complete':True,'completed':len(rows),'total':20,
          'labels':LABELS,'methods':ORDER,'baseline':'s10','sites':SITES,'site_colors':COLORS,
          'site_session_blocks':{s:len(bs) for s,bs in blocks.items()},'bootstrap_replicates':5000,'bootstrap_seed':20261002,
          'summary':summary,'paired_deltas':deltas,'k_sensitivity':sensitivity,'seam_sensitivity':seam_sensitivity,'scale_addition_contrasts':additions,
          'images':rows,'selection':selection['selection'],'selection_seed':selection['seed'],
          'protocol':json.loads((out/'protocol.json').read_text(encoding='utf-8'))}
    save(out/'results.json',data)
    return data


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    rows=data['images']; jitter=np.linspace(-.24,.24,len(rows))
    rng=np.random.default_rng(54);rng.shuffle(jitter)
    legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[s],markersize=6,label=SITES[s]) for s in SITES]
    legend.append(Line2D([0],[0],marker='D',color='#192d32',markersize=6,label='Mean and exploratory 95% interval'))
    def dots(ax,metric,delta=False,show_labels=True):
        for j,m in enumerate(ORDER):
            v=np.array([val(r,m,metric) for r in rows])
            if delta:v-=np.array([val(r,'s10',metric) for r in rows])
            ax.scatter(v,np.full(len(rows),j)+jitter,c=[COLORS[r['site']] for r in rows],s=24,alpha=.76,edgecolors='white',linewidths=.25,zorder=3)
            q=data['paired_deltas' if delta else 'summary'][metric][m];mean=q['mean'];lo,hi=q['ci95']
            ax.hlines(j,lo,hi,color='#152e35',lw=2.4,zorder=4)
            ax.plot(mean,j,'D',color='#152e35',markersize=6,zorder=5)
        ax.set_yticks(range(len(ORDER)),[LABELS[m]+('  [baseline]' if m=='s10' else '') for m in ORDER] if show_labels else [])
        ax.set_ylim(len(ORDER)-.55,-.55);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        if delta:ax.axvline(0,color='#849696',ls='--',lw=1)
    fig,axes=plt.subplots(1,2,figsize=(14,7.6),gridspec_kw={'width_ratios':[1,1]})
    dots(axes[0],'f1');dots(axes[1],'f1',True,False)
    axes[0].set_title('RGB-edge F1 for each photograph',loc='left',weight='bold',pad=16)
    axes[1].set_title('Paired change from 10% four-flip baseline',loc='left',weight='bold',pad=16)
    axes[0].set_xlabel('RGB-edge F1 (K = 6); higher = closer edge alignment')
    axes[1].set_xlabel('Change in F1 (treatment minus baseline)')
    axes[0].set_xlim(0,1)
    fig.suptitle('Multiscale DINOv3 · 20 photographs · four-flip averaging at every scale',x=.025,ha='left',fontsize=16,weight='bold')
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.055),ncol=3,frameon=False)
    fig.text(.025,.018,'Dots = photographs; diamonds = means. Intervals: paired site/session-block bootstrap, stratified by site (12 blocks).\nBalanced rank-stratified pilot; image-edge agreement is not leaf-segmentation accuracy.',fontsize=9,color='#4a6265')
    fig.tight_layout(rect=(0,.14,1,.94),w_pad=3)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'score_comparison.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,12))
    specs=[('precision','RGB-edge precision','Fraction of predicted boundaries near an RGB edge'),
           ('recall','RGB-edge recall','Fraction of RGB edges near a predicted boundary'),
           ('seam_distance','Crop-border diagnostic |ratio - 1|','10% test grid; smaller = less measured border contrast'),
           ('boundary_density','Boundary density','Fraction of eligible pixels labeled as boundary; descriptive')]
    for ax,(metric,title,label) in zip(axes.flat,specs):
        dots(ax,metric);ax.set_title(title,loc='left',weight='bold',pad=12);ax.set_xlabel(label)
    fig.suptitle('Why scores change · complementary diagnostics (K = 6)',x=.025,ha='left',fontsize=17,weight='bold')
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.02),ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.095,1,.95),w_pad=3,h_pad=3)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'diagnostics.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,7.6))
    for offset,(k,col) in zip([-.18,0,.18],[(4,'#d18d38'),(6,'#286a72'),(8,'#8664a8')]):
        for j,m in enumerate(ORDER):
            q=data['k_sensitivity'][str(k)][m]['delta'];lo,hi=q['ci95']
            axes[0].hlines(j+offset,lo,hi,color=col,lw=1.5)
            axes[0].plot(q['mean'],j+offset,'o',color=col,ms=5,label=f'K = {k}' if j==0 else None)
    axes[0].set_yticks(range(9),[LABELS[m] for m in ORDER]);axes[0].set_ylim(8.6,-.6)
    axes[0].axvline(0,color='#899a9c',ls='--',lw=1);axes[0].grid(axis='x',alpha=.2)
    axes[0].set_xlabel('Mean paired F1 change vs 10% baseline at the same K')
    axes[0].set_title('Sensitivity to number of clusters',loc='left',weight='bold',pad=15)
    axes[0].legend(frameon=False,loc='lower left')
    for j,m in enumerate(ORDER):
        x=data['summary']['encoding_cost_ratio'][m]['mean'];q=data['summary']['f1'][m];lo,hi=q['ci95']
        axes[1].vlines(x,lo,hi,color='#718f8b',lw=1.5)
        axes[1].scatter(x,q['mean'],s=64,color='#235f66' if m!='s10' else '#b16a32',zorder=3)
        # Numbered points avoid colliding treatment labels.
        axes[1].annotate(str(j+1),(x,q['mean']),xytext=(5,5),textcoords='offset points',weight='bold',fontsize=9)
    axes[1].set_xlabel('Estimated encoding cost / 10% encoding cost')
    axes[1].set_ylabel('Mean RGB-edge F1 (K = 6)');axes[1].grid(alpha=.2)
    axes[1].set_title('Edge agreement versus computation',loc='left',weight='bold',pad=15)
    axes[1].text(.98,.03,'\n'.join(f'{j+1}. {LABELS[m]}' for j,m in enumerate(ORDER)),transform=axes[1].transAxes,ha='right',va='bottom',fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.suptitle('Robustness and computational cost',x=.025,ha='left',fontsize=17,weight='bold')
    fig.text(.025,.025,'All four orientations are included in every treatment. Mixture cost is the sum of measured constituent-scale encoding times;\nprojection, blending and clustering are excluded. Intervals are exploratory; no multiplicity adjustment.',fontsize=9,color='#4a6265')
    fig.tight_layout(rect=(0,.085,1,.94),w_pad=3)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'robustness_cost.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6.5))
    for j,q in enumerate(data['scale_addition_contrasts']):
        v=np.array([val(r,q['to'],'f1')-val(r,q['from'],'f1') for r in rows])
        ax.scatter(v,np.full(len(rows),j)+jitter,c=[COLORS[r['site']] for r in rows],s=26,alpha=.75,zorder=3)
        ax.hlines(j,*q['ci95'],color='#152e35',lw=2.4,zorder=4)
        ax.plot(q['mean'],j,'D',color='#152e35',markersize=6,zorder=5)
    ax.set_yticks(range(len(ADDITIONS)),[f'{LABELS[a]}  →  {LABELS[b]}' for a,b in ADDITIONS]);ax.set_ylim(len(ADDITIONS)-.55,-.55)
    ax.axvline(0,color='#899a9c',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Paired change in RGB-edge F1 (after minus before; K = 6)')
    fig.suptitle('Does adding a scale help?',x=.025,ha='left',fontsize=17,weight='bold')
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.055),ncol=3,frameon=False)
    fig.text(.025,.018,'Adding a scale reweights all included scales equally. These contrasts distinguish averaging effects from choosing a finer single scale.',fontsize=9,color='#4a6265')
    fig.tight_layout(rect=(0,.17,1,.92))
    for ext in ['png','svg','pdf']:fig.savefig(out/f'scale_additions.{ext}',dpi=190,facecolor='white')
    plt.close(fig)


def tables(out,data):
    with (out/'image_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['id','site','session','prior_visual_rank','selection_quintile','treatment','K',*METRICS,'estimated_encoding_seconds','crop_inferences','seam_ratio_grid5','seam_ratio_grid10','seam_ratio_grid20','seam_ratio_grid40'])
        for r in data['images']:
            for m in ORDER:
                q=r['methods'][m]
                for k in [4,6,8]: w.writerow([r['id'],r['site'],r['session'],r['prior_visual_rank'],r['selection_quintile'],LABELS[m],k,*[val(r,m,x,k) for x in METRICS],q['estimated_encoding_seconds'],q['crop_inferences'],*[q['seam_grids'][str(g)]['ratio'] for g in [5,10,20,40]]])
    with (out/'treatment_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['treatment','metric','mean','ci_low','ci_high','mean_paired_change','paired_ci_low','paired_ci_high'])
        for m in ORDER:
            for metric in METRICS:
                q=data['summary'][metric][m];d=data['paired_deltas'][metric][m]
                w.writerow([LABELS[m],metric,q['mean'],*q['ci95'],d['mean'],*d['ci95']])


def build_html(out,data):
    means=data['summary']['f1'];deltas=data['paired_deltas']['f1']
    best=max(ORDER,key=lambda m:means[m]['mean']); best_mix=max([m for m in ORDER if '_' in m],key=lambda m:means[m]['mean'])
    all4='s05_10_20_40';qq=deltas[all4]
    findings=(f"The 10% baseline scored {means['s10']['mean']:.3f}. The equal-weight 5% + 10% + 20% + 40% mixture scored {means[all4]['mean']:.3f} "
              f"(paired change {qq['mean']:+.3f}; exploratory 95% interval {qq['ci95'][0]:+.3f} to {qq['ci95'][1]:+.3f}), "
              f"with higher scores in {qq['positive']} of 20 photos. The highest mean in this pilot was {LABELS[best]} ({means[best]['mean']:.3f}); "
              f"the highest-scoring mixture was {LABELS[best_mix]} ({means[best_mix]['mean']:.3f}).")
    summary_rows=''
    for m in ORDER:
        q=means[m];d=deltas[m];s=data['summary']
        summary_rows+=f'<tr class="{"baseline" if m=="s10" else ""}"><th>{LABELS[m]}{" · baseline" if m=="s10" else ""}</th><td><b>{q["mean"]:.3f}</b><small>{q["ci95"][0]:.3f}–{q["ci95"][1]:.3f}</small></td><td>{d["mean"]:+.3f}<small>{d["ci95"][0]:+.3f} to {d["ci95"][1]:+.3f}</small></td><td>{d["positive"]}/20</td><td>{s["precision"][m]["mean"]:.3f}</td><td>{s["recall"][m]["mean"]:.3f}</td><td>{s["seam_distance"][m]["mean"]:.4f}</td><td>{s["encoding_cost_ratio"][m]["mean"]:.2f}×</td></tr>'
    site_rows=''
    for site in SITES:
        site_rows+=f'<tr><th>{SITES[site]} (n=5)</th>'+''.join(f'<td>{means[m]["site_means"][site]:.3f}<small>Δ {deltas[m]["site_means"][site]:+.3f}</small></td>' for m in ORDER)+'</tr>'
    seam_rows=''
    for m in ORDER:
        seam_rows+=f'<tr><th>{LABELS[m]}</th>'+''.join(f'<td>{data["seam_sensitivity"][str(g)][m]["score"]["mean"]:.4f}<small>Δ {data["seam_sensitivity"][str(g)][m]["delta"]["mean"]:+.4f}</small></td>' for g in [5,10,20,40])+'</tr>'
    selection_rows=''
    for r in data['images']:
        selection_rows+=f'<tr><th>{r["id"]}</th><td>{SITES[r["site"]]}</td><td>{html.escape(r["session"])}</td><td>{r["selection_quintile"]}</td><td>{r["prior_visual_rank"]}</td><td>{r["size"][0]} × {r["size"][1]}</td></tr>'
    norm_rows=''
    for s in [5,10,20,40]:
        norm=np.mean([r['scales'][str(s)]['mean_raw_fourflip_token_norm'] for r in data['images']])
        sd=np.mean([r['scales'][str(s)]['projected_spatial_sd'] for r in data['images']])
        tiles=sum(r['scales'][str(s)]['tiles'] for r in data['images'])
        norm_rows+=f'<tr><th>{s}%</th><td>{norm:.2f}</td><td>{sd:.3f}</td><td>{tiles:,}</td></tr>'
    checks={'baseline_min_label_ari':min(r['checks']['baseline_label_ari'] for r in data['images']),
            'baseline_max_abs_f1_delta':max(abs(v) for r in data['images'] for v in r['checks']['baseline_f1_delta'].values()),
            'baseline_max_abs_seam_delta':max(abs(r['checks']['baseline_seam_delta']) for r in data['images']),
            'all_20_baselines_passed':True,'images':20,'treatments':9,'K_settings':3,'metric_combinations':540}
    save(out/'validation.json',checks)
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    addition_rows=''
    for q in data['scale_addition_contrasts']:
        addition_rows+=f'<tr><th>{LABELS[q["from"]]} → {LABELS[q["to"]]}</th><td>{q["mean"]:+.3f}</td><td>{q["ci95"][0]:+.3f} to {q["ci95"][1]:+.3f}</td><td>{q["positive"]}/20</td></tr>'
    source=HTML.replace('__DATA__',payload).replace('__FINDINGS__',html.escape(findings)).replace('__SUMMARY_ROWS__',summary_rows).replace('__SITE_ROWS__',site_rows).replace('__METHOD_HEADS__',''.join(f'<th>{LABELS[m]}</th>' for m in ORDER)).replace('__SEAM_ROWS__',seam_rows).replace('__SELECTION_ROWS__',selection_rows).replace('__NORM_ROWS__',norm_rows).replace('__ADDITION_ROWS__',addition_rows).replace('__VALIDATION__',html.escape(json.dumps(checks,indent=2)))
    (out/'index.html').write_text(source,encoding='utf-8')
    (out/'findings.txt').write_text(findings+'\n\n'+json.dumps({'f1_means':means,'paired_f1_changes':deltas,'validation':checks},indent=2,ensure_ascii=False),encoding='utf-8')
    print(findings)
    print(json.dumps(checks))


HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv3 · Multiscale four-flip pilot</title>
<style>
:root{--ink:#173c40;--muted:#586f71;--line:#d4dfda;--paper:#f3f5ef;--accent:#267f70}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:34px 28px 80px}h1{font-size:clamp(30px,4vw,48px);line-height:1.12;max-width:1100px;margin:12px 0 22px}h2{font-size:26px;margin:0 0 14px}h3{font-size:18px;margin:0 0 12px}p{max-width:1100px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:700;color:var(--accent)}.lead{font-size:19px;line-height:1.6}.subtle,small{color:var(--muted)}small{display:block;font-size:12px;white-space:nowrap}.badges{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}.badge{border:1px solid #bad1c6;padding:5px 13px;border-radius:30px;background:#e9f2e9;font-size:14px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}a{color:#167063;text-underline-offset:3px}section{background:#fff;border:1px solid var(--line);border-radius:18px;padding:26px;margin:24px 0;scroll-margin-top:16px}.chart{width:100%;height:auto;display:block}.scroll{overflow:auto;max-width:100%}table{border-collapse:collapse;min-width:100%;font-size:14px}th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}thead th{font-size:12px;color:var(--muted);background:#f7f9f4}.baseline{background:#edf5ee}details{margin-top:18px}summary{cursor:pointer;color:#23685d;font-weight:600;padding:8px 0}.note{background:#f3f7f1;border-left:3px solid #a2c3b4;padding:14px 18px}.downloads{display:flex;gap:16px;font-size:14px;flex-wrap:wrap;margin-top:12px}.controls{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:14px 0 22px}label{display:block;font-size:13px;color:var(--muted)}select,button{font:inherit;padding:9px 12px;color:var(--ink);background:white;border:1px solid #bdccc3;border-radius:8px;max-width:100%}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.photo-context{display:flex;align-items:center;gap:24px;margin-bottom:22px;background:#f5f7f2;padding:16px;border-radius:12px}.photo-context img{width:200px;max-height:180px;object-fit:contain}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.tile{border:1px solid var(--line);padding:14px;border-radius:12px}.tile h3{font-size:16px;display:flex;justify-content:space-between;gap:8px}.tile img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#edf0e9;display:block}.tile a{display:block}.tile .stats{font-size:13px;margin-top:10px;display:flex;justify-content:space-between;gap:8px}.caption{font-size:14px;color:var(--muted)}pre{white-space:pre-wrap;font-size:13px;word-break:break-word}.formula{font-family:Georgia,serif;font-size:21px;background:#f4f7f2;padding:15px}.foot{font-size:13px;color:var(--muted)}@media(max-width:850px){main{padding:22px 14px 50px}section{padding:18px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.photo-context{align-items:start}.photo-context img{width:150px}}@media(max-width:540px){.gallery{grid-template-columns:1fr}.photo-context{display:block}.photo-context img{width:100%;height:200px}h2{font-size:22px}.controls{display:block}.controls>div,.controls>button{margin:8px 0}.lead{font-size:17px}}
</style></head><body><main>
<div class="eyebrow">DINOv3 · matched multiscale experiment · 01 October 2026</div>
<h1>Does combining crop scales improve the benchmark?</h1>
<p class="lead">__FINDINGS__</p>
<div class="badges"><span class="badge">20 photos · 5 per site</span><span class="badge">9 treatments</span><span class="badge">Four flips at every scale</span><span class="badge">Fixed PCA · separate cluster fits</span></div>
<p class="note"><b>Interpretation:</b> this experiment measures agreement with RGB image edges. It does not measure leaf IoU, species recognition, or biological diversity. A higher F1 can arise from finer texture segmentation. Treatment differences are exploratory, based on a small balanced pilot selected across the prior diversity rankings.</p>
<nav><a href="#scores">Scores and paired changes</a><a href="#additions">Adding scales</a><a href="#diagnostics">Diagnostics</a><a href="#robustness">Robustness and cost</a><a href="#sites">Four sites</a><a href="#gallery">Inspect the photos</a><a href="#methods">Methods and checks</a></nav>
<section id="scores"><h2>Every photo, plus the mean</h2>
<p>Each colored point represents one photo; the same 20 photos appear in every treatment. Dark diamonds show means with exploratory 95% intervals. The right panel subtracts each photo’s 10% four-flip baseline, so the comparison is paired. Positive changes indicate greater RGB-edge agreement.</p>
<img class="chart" src="score_comparison.png" alt="Twenty individual image scores per treatment, means and intervals; paired score changes from the 10 percent baseline">
<div class="downloads"><a href="score_comparison.svg">Vector figure (SVG)</a><a href="score_comparison.pdf">Print figure (PDF)</a><a href="image_metrics.csv">All photo scores (CSV)</a><a href="treatment_summary.csv">Means and paired changes (CSV)</a></div>
<details open><summary>Numeric comparison · K = 6</summary><div class="scroll"><table><thead><tr><th>Crop scale(s)</th><th>Mean F1<br>95% interval</th><th>Paired Δ F1<br>95% interval</th><th>Photos<br>improved</th><th>Precision</th><th>Recall</th><th>Seam<br>|ratio−1|</th><th>Encoding<br>cost</th></tr></thead><tbody>__SUMMARY_ROWS__</tbody></table></div><p class="caption">Means weight each photo equally; five photos per site also gives equal site weight. The baseline is identical to itself, so its change is zero. Cost is the mean of within-photo encoding-time ratios.</p></details></section>
<section id="additions"><h2>Does averaging help beyond choosing a finer scale?</h2><p>A multiscale treatment can beat 10% simply because it includes a finer 5% representation. These paired comparisons test what changes when another scale is added. All included scales are reweighted equally, and clusters are fitted again.</p><img class="chart" src="scale_additions.png" loading="lazy" alt="Individual paired F1 changes and means for adding scales to single-scale and multiscale treatments"><div class="downloads"><a href="scale_additions.svg">SVG</a><a href="scale_additions.pdf">PDF</a></div><details><summary>Scale-addition values</summary><div class="scroll"><table><thead><tr><th>Before → after</th><th>Mean Δ F1</th><th>Exploratory 95% interval</th><th>Photos improved</th></tr></thead><tbody>__ADDITION_ROWS__</tbody></table></div></details></section>
<section id="diagnostics"><h2>What contributes to the score?</h2><p>F1 balances boundary precision and recall. Boundary density helps reveal whether a treatment adds more boundaries. The seam diagnostic compares feature jumps at crop borders with off-border jumps matched for RGB contrast; all treatments use the same 10% test grid here.</p>
<img class="chart" src="diagnostics.png" loading="lazy" alt="Individual points and mean intervals for precision, recall, seam contrast and boundary density">
<div class="downloads"><a href="diagnostics.svg">SVG</a><a href="diagnostics.pdf">PDF</a></div>
<details><summary>Seam diagnostic at all four test grids</summary><p>The table gives mean |ratio−1| and its paired change from the 10% treatment. Each column applies an identical lattice to all treatments. Smaller values mean less measured border contrast, but this diagnostic can miss visually apparent grid artifacts; it is not a validated lattice detector.</p><div class="scroll"><table><thead><tr><th>Treatment</th><th>5% test grid</th><th>10% test grid</th><th>20% test grid</th><th>40% test grid</th></tr></thead><tbody>__SEAM_ROWS__</tbody></table></div></details></section>
<section id="robustness"><h2>Cluster-count sensitivity and computation</h2><p>K = 6 is the primary setting fixed before this pilot. K = 4 and K = 8 check whether the direction of the comparison depends on the number of clusters. Every K has independently fitted centers for each treatment.</p>
<img class="chart" src="robustness_cost.png" loading="lazy" alt="Mean paired F1 changes at K4 K6 K8 and mean F1 versus estimated encoding cost">
<div class="downloads"><a href="robustness_cost.svg">SVG</a><a href="robustness_cost.pdf">PDF</a></div><p class="caption">Encoding cost sums the measured per-scale encoding times used by a treatment. It is an estimate of running that treatment alone; these fields were reused across mixtures in the actual experiment. CPU projection, blending, clustering, file I/O and report generation are excluded.</p></section>
<section id="sites"><h2>Do the four sites behave similarly?</h2><p>Mean RGB-edge F1 and paired change from the 10% treatment. Each site contributes five photos; these small subsets do not establish site-wide performance.</p><div class="scroll"><table><thead><tr><th>Site</th>__METHOD_HEADS__</tr></thead><tbody>__SITE_ROWS__</tbody></table></div></section>
<section id="gallery"><h2>Inspect all nine treatments on the same photo</h2><p>PCA colors share a frozen basis and display range within each photo. Cluster-overlay colors are assigned independently and do not imply matched biological groups. Click an image to open its preview.</p>
<div class="controls"><div><label for="photo">Photo · prior diversity rank</label><select id="photo"></select></div><div><label for="view">Display</label><select id="view"><option value="pca">PCA colors</option><option value="overlay">K = 6 cluster overlay</option><option value="boundary">K = 6 boundaries on RGB</option></select></div><button id="prev">← Previous</button><button id="next">Next →</button></div>
<div id="photo-context" class="photo-context"></div><div id="treatments" class="gallery"></div><p class="caption">Magenta in boundary mode marks predicted cluster boundaries. Gray margins are the inherited artifact-exclusion/evaluation mask. No vegetation-only mask and no AnyUp are used in this comparison.</p></section>
<section id="methods"><h2>Methods, sampling and reproducibility</h2>
<p><b>Sampling:</b> the 439 unique source photos were separated by site and sorted by the existing whole-image visual-diversity rank. One photo was drawn from each of five nearly equal rank groups within each site, using a fixed seed (20261001). This gives five photos per site, selected before examining the new multiscale outcomes. These are exploratory development comparisons, not a held-out validation set.</p>
<p><b>Input:</b> existing cleaned RGB working images, longest side at most 1280 pixels. Square crop sides are 5%, 10%, 20% or 40% of the shorter image dimension, rounded to pixels. These are side-length percentages, not area percentages. Every crop is resized bicubically to 384 × 384. Crop stride is two thirds of crop width. DINOv3 ViT-B/16 uses the same pinned model revision and RGB /255 plus processor mean/std as the previous benchmark.</p>
<p><b>Four-flip extraction:</b> identity, horizontal, vertical and combined reflections of each crop are encoded. The 24 × 24 token grids are reflected back to the original orientation and their raw 768-dimensional vectors averaged. No new L2 normalization is introduced. The original Phase 1 PCA mean and 16-component basis are frozen per photo; token grids are projected, bilinearly interpolated, and blended with the existing Hann weights. Each scale’s overlapping crops are normalized separately.</p>
<div class="formula">F<sub>mixture</sub>(x,y) = (1 / |S|) ∑<sub>s∈S</sub> F<sub>s</sub>(x,y)</div>
<p><b>Scale fusion:</b> aligned feature fields receive equal coefficients, regardless of crop count. Because the same centered PCA projection is affine and weights sum to one, averaging in that fixed 16-dimensional space is equivalent to averaging raw features first and then projecting, up to floating-point error. This does not mean that separately fitted PCA color images or cluster IDs can be averaged. Raw feature magnitudes can differ between scales; equal coefficients are not an equal-variance constraint.</p>
<details><summary>Feature magnitude and crop counts</summary><p>Each norm/SD is first measured within a photo and then averaged across the 20 photos. Raw token norms follow four-flip averaging. Spatial SD is the RMS across 16 component standard deviations at the frozen sample coordinates. Crop counts exclude the factor of four for flips.</p><div class="scroll"><table><thead><tr><th>Scale</th><th>Mean raw token norm</th><th>Projected spatial SD</th><th>Total crops · 20 photos</th></tr></thead><tbody>__NORM_ROWS__</tbody></table></div></details>
<p><b>Clustering:</b> separate K-means fits for every treatment, K = 4, 6 and 8, n_init = 10, seed = 42. Training uses the same 4096 eligible spatial coordinates as the previous benchmark. All metrics use the same inherited artifact mask and 16-pixel erosion. Shared cluster centers and vegetation scoring masks are excluded. The frozen PCA was originally fitted to the Phase 1 whole-image, 10% native and 10% enlarged representations; this pilot tests new scales in that existing feature subspace, rather than refitting PCA for them.</p>
<p><b>RGB-edge F1:</b> convert RGB to weighted grayscale, smooth with a Gaussian (σ = 1 pixel), compute Sobel gradient magnitude, and select the strongest 15% of eligible gradients. Precision is the fraction of predicted boundary pixels within two pixels of this reference; recall reverses that comparison. F1 = 2PR/(P + R). The tolerance uses two iterations of SciPy binary dilation with its default cross-shaped structuring element (Manhattan distance ≤ 2). Image texture, shadows, soil, stems and leaf veins can all contribute reference edges.</p>
<p><b>Uncertainty:</b> 5000 paired bootstrap draws (seed 20261002) resample the observed date/session groups within each site and average the four site means with equal weights. There are 12 observed site/session blocks (3 Feßbach, 3 Heidesheim, 4 Rettmer, 2 Ribbeck). Intervals describe sensitivity to these sampled groups; rank-stratum sampling and repeated physical plots across dates are not fully modeled. The nine treatments are exploratory comparisons with no multiplicity correction.</p>
<details><summary>All 20 selected photos</summary><div class="scroll"><table><thead><tr><th>Photo</th><th>Site</th><th>Session</th><th>Within-site rank quintile</th><th>Prior global visual rank</th><th>Working dimensions</th></tr></thead><tbody>__SELECTION_ROWS__</tbody></table></div></details>
<details><summary>Control checks and downloadable provenance</summary><p>The recomputed 10% treatment was checked against the earlier four-flip run. Label comparison uses adjusted Rand index, which ignores arbitrary cluster-number permutations. All 20 controls passed.</p><pre>__VALIDATION__</pre><div class="downloads"><a href="selection.json">Frozen selection</a><a href="protocol.json">Protocol</a><a href="results.json">Full results</a><a href="validation.json">Control checks</a><a href="image_metrics.csv">Photo metrics</a><a href="analysis_code.zip">Analysis scripts and frozen protocol</a></div></details>
</section><p class="foot">Multiscale pilot · existing taxonomy and full-dataset diversity reports are unchanged. Dense float32 feature fields and K = 6 label maps are retained with the server run for reproducibility.</p>
</main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),photos=D.images,sel=document.getElementById('photo'),view=document.getElementById('view');
for(const [i,r] of photos.entries()){const o=document.createElement('option');o.value=i;o.textContent=`${r.id} · ${D.sites[r.site]} · prior rank ${r.prior_visual_rank}`;sel.append(o)}
const signed=x=>(x>=0?'+':'')+x.toFixed(3);
function render(){const i=Number(sel.value),r=photos[i],baseline=r.methods.s10.k['6'].rgb_edge.f1;document.getElementById('prev').disabled=i===0;document.getElementById('next').disabled=i===photos.length-1;
document.getElementById('photo-context').innerHTML=`<a href="${r.id}/original.jpg" target="_blank"><img src="${r.id}/original.jpg" alt="Original RGB ${r.id}"></a><div><h3>${r.id} · ${D.sites[r.site]}</h3><p>Session: ${r.session}<br>Prior visual-diversity rank: ${r.prior_visual_rank} / 439<br>Within-site rank quintile: ${r.selection_quintile} / 5<br>10% baseline F1: <b>${baseline.toFixed(3)}</b></p><a href="${r.id}/native_center_comparison.jpg" target="_blank">Compare all treatments at native working-pixel resolution ↗</a></div>`;
document.getElementById('treatments').innerHTML=D.methods.map(m=>{const q=r.methods[m],f=q.k['6'].rgb_edge.f1;return `<article class="tile ${m==='s10'?'baseline':''}"><h3>${D.labels[m]}${m==='s10'?'<span>Baseline</span>':''}</h3><a href="${r.id}/${m}_${view.value}.jpg" target="_blank"><img loading="lazy" src="${r.id}/${m}_${view.value}.jpg" alt="${D.labels[m]} ${view.value} ${r.id}"></a><div class="stats"><span>F1 <b>${f.toFixed(3)}</b></span><span>Δ ${signed(f-baseline)}</span><span>${(q.estimated_encoding_seconds/r.methods.s10.estimated_encoding_seconds).toFixed(2)}× cost</span></div></article>`}).join('');}
sel.addEventListener('change',render);view.addEventListener('change',render);document.getElementById('prev').onclick=()=>{sel.value=Number(sel.value)-1;render()};document.getElementById('next').onclick=()=>{sel.value=Number(sel.value)+1;render()};render();
</script></body></html>'''


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--no-figures',action='store_true');args=ap.parse_args()
    data=aggregate(args.out)
    tables(args.out,data)
    if not args.no_figures:figures(args.out,data)
    with zipfile.ZipFile(args.out/'analysis_code.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['multiscale_benchmark.py','phase1_benchmark.py','phase1_fourflip_benchmark.py','build_multiscale_report.py','inspect_multiscale_pilot.py']:
            z.write(Path(__file__).with_name(name),arcname=name)
        for name in ['selection.json','protocol.json']:z.write(args.out/name,arcname=name)
    build_html(args.out,data)
