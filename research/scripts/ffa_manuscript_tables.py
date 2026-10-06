"""Generate all numeric manuscript tables directly from the audited JSON."""
import json,re
from pathlib import Path
root=Path('outputs/forced-focused-attention-preprint');e=json.loads((root/'data/evidence.json').read_text(encoding='utf-8'))
names={'moin':'Grassland (MOIN)','fungal_network':'Fungal networks','neon':'Airborne forest (NEON)','coralscapes':'Coral reefs (Coralscapes)','pmid':'Phytoplankton (PMID2019)'}
methods=['whole','sam3','s05','s10']
rows=[]
for d,z in e['domains'].items():
    vals=[f"{z['rgb'][m]['mean']:.3f} $\\pm$ {z['rgb'][m]['sd']:.3f}" for m in methods]
    rows.append(names[d]+' & '+' & '.join(vals)+r' \\')
(root/'table_f1.tex').write_text('\n'.join(rows),encoding='utf-8')
rows=[]
z=e['cost']['whole']
rows.append('DINOv3 & '+f"{z['seconds']['mean']:.2f} $\\pm$ {z['seconds']['sd']:.2f} & -- & --"+r' \\')
for m,label in [('sam3','SAM3 automatic'),('s05','FFA 5\\%'),('s10','FFA 10\\%')]:
    z=e['cost'][m];rows.append(label+' & '+f"{z['seconds']['mean']:.1f} $\\pm$ {z['seconds']['sd']:.1f} & {z['gpu_board_joules']['mean']/1000:.2f} $\\pm$ {z['gpu_board_joules']['sd']/1000:.2f} & {z['inverse_joules']['mean']*1000:.3f} $\\pm$ {z['inverse_joules']['sd']*1000:.3f}"+r' \\')
(root/'table_cost.tex').write_text('\n'.join(rows),encoding='utf-8')
rows=[]
for z in e['contrasts']:
    f='5\\%' if z['focus']=='s05' else '10\\%';base='DINOv3' if z['baseline']=='whole' else 'SAM3'
    rows.append(names[z['dataset']]+' & '+f+' & '+base+' & '+f"{z['mean_delta']:.3f} & [{z['ci95'][0]:.3f}, {z['ci95'][1]:.3f}] & {z['wins']}/10 & {z['exact_two_sided_p']:.5f} & {z['holm_p']:.5f}"+r' \\')
(root/'table_contrasts.tex').write_text('\n'.join(rows),encoding='utf-8')
rows=[]
for lam in [1,2,4,8,16,32,64]:
    z=[next(z for z in e['clustering']['distributions'] if z['dataset']=='all' and z['method']==m and z['dimensions']==16 and z['algorithm']=='penalty' and z['setting']==lam) for m in ['s05','s10']]
    rows.append(str(lam)+' & '+' & '.join(f"{x['mean_k']:.2f} & {x['median_k']:.0f} & {100*x['preferred_4_6_fraction']:.0f}\\% & {100*x['upper_12_fraction']:.0f}\\%" for x in z)+r' \\')
(root/'table_penalty.tex').write_text('\n'.join(rows),encoding='utf-8')
rows=[]
for d in [3,16]:
    for m in ['whole','s05','s10']:
        z=next(z for z in e['clustering']['distributions'] if z['dataset']=='all' and z['method']==m and z['dimensions']==d and z['algorithm']=='penalty' and z['setting']==4)
        rows.append(str(d)+' & '+{'whole':'DINOv3','s05':'FFA 5\\%','s10':'FFA 10\\%'}[m]+' & '+f"{z['mean_k']:.2f} & {z['median_k']:.0f} & {100*z['preferred_4_6_fraction']:.0f}\\% & {100*z['upper_12_fraction']:.0f}\\%"+r' \\')
(root/'table_dimensions.tex').write_text('\n'.join(rows),encoding='utf-8')
main=root/'main.tex'
if main.exists():
    text=main.read_text(encoding='utf-8')
    for name in ['f1','cost','contrasts','penalty','dimensions']:
        block='% BEGIN GENERATED TABLE '+name+'\n'+(root/f'table_{name}.tex').read_text(encoding='utf-8')+'\n% END GENERATED TABLE '+name
        token='\\input{table_'+name+'.tex}'
        if token in text:text=text.replace(token,block)
        else:text=re.sub('% BEGIN GENERATED TABLE '+name+r'\n.*?% END GENERATED TABLE '+name,lambda m:block,text,flags=re.S)
    main.write_text(text,encoding='utf-8')
print('Generated five tables from',e['source_sha256'])
