"""Independent saved-result checks: source integrity, coverage, metric recomputation."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,distance_transform_cdt
from sklearn.metrics import adjusted_rand_score
R=Path('/home/masahiro/kics-zert2/runs/eco_collection75_v1')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edge(lab):
    out=np.zeros(lab.shape,bool);out[:,:-1]=lab[:,:-1]!=lab[:,1:];out[:-1,:]|=lab[:-1,:]!=lab[1:,:];return out
def boundary_f1(a,b,valid,tol):
    a=a&valid;b=b&valid
    pr=float((distance_transform_cdt(~b,metric='taxicab')[a]<=tol).mean()) if a.any() and b.any() else 0
    rec=float((distance_transform_cdt(~a,metric='taxicab')[b]<=tol).mean()) if a.any() and b.any() else 0
    return 2*pr*rec/(pr+rec) if pr+rec else 0
rows=[]
for d in ['moin','mycelium','fungal_network','bam','neon','coralscapes','tara','pmid']:
    rr=json.loads((R/f'{d}_selection.json').read_text())['images'];assert len(rr)==10;rows+=rr
assert len({r['source_sha256'] for r in rows})==80
checks=0;complete=True;counts={};references=0
for r in rows:
    dest=R/r['id'];im=Image.open(dest/'input.png');w,h=im.size
    assert [w,h]==r['working_dimensions'] and min(w,h)>=1024 and w%16==h%16==0 and r['resize_ratio']==1
    assert digest(dest/'input.png')==r['input_sha256'];assert digest(Path(r['original_path']))==r['source_sha256']
    x0,y0,x1,y1=r['native_window'];assert x1-x0==w and y1-y0==h
    gt=np.asarray(Image.open(dest/'reference.png')) if r['reference_available'] else None
    valid=binary_erosion(gt>0 if gt is not None and r['dataset']=='coralscapes' else np.ones((h,w),bool),iterations=2,border_value=0)
    if gt is not None:assert gt.shape==(h,w) and digest(dest/'reference.png')==r['reference_sha256']
    f=dest/'dino_metrics.json';dino=json.loads(f.read_text())['methods'] if f.exists() else {};counts[r['id']]=len(dino)
    complete&=set(dino)=={'whole','whole_fourflip','s05','s10','s20','s40'}
    for name,q in dino.items():
        e=q['extraction'];assert e['seconds']>0 and (e['gpu_board_joules'] is None or e['gpu_board_joules']>0)
        if name.startswith('s'):
            c=e['crop_pixels'];s=e['stride_pixels'];assert c==max(16,round(min(w,h)*int(name[1:])/100)) and s==round(c/4)
            xs=sorted(set(range(0,w-c+1,s))|{w-c});ys=sorted(set(range(0,h-c+1,s))|{h-c});assert len(xs)*len(ys)==e['tiles']
            assert xs[0]==ys[0]==0 and xs[-1]+c==w and ys[-1]+c==h and max(np.diff(xs))<=c and max(np.diff(ys))<=c
        for k in [6,12]:
            lab=np.load(dest/f'{name}_k{k}_labels.npz')['labels'];assert lab.shape==(h,w) and lab.min()>=0 and lab.max()<k
            v=q['k'][str(k)];assert 0<=v['rgb_edge']['f1']<=1
            if gt is not None and r['id'].endswith('_01'):
                for tol,key in [(2,'boundary'),(5,'boundary_5px')]:
                    ref=v['reference'][key]
                    if ref:assert abs(boundary_f1(edge(lab),edge(gt),valid,tol)-ref['f1'])<1e-12;references+=1
            checks+=1
    f=dest/'sam3_hf_metrics.json';complete&=f.exists()
    if f.exists():
        q=json.loads(f.read_text());counts[r['id']]+=1;pro=np.load(dest/'sam3_hf_proposals.npz');assert len(pro['quality'])==q['extraction']['mask_count'];assert np.all(pro['quality']>.88);assert q['extraction']['user_supplied_prompt_count']==0 and q['extraction']['implementation']=='hf_mask_generation'
        for variant in ['quality','small_first']:
            lab=np.load(dest/f'sam3_hf_{variant}_labels.npz')['labels'];assert lab.shape==(h,w)
            if r['id'].endswith('_01'):
                z=np.zeros((h,w),np.uint16);order=np.argsort(pro['quality'],kind='stable') if variant=='quality' else np.argsort(-pro['area'],kind='stable')
                for j in order:z[np.unpackbits(pro['packed'][j],count=w*h).reshape(h,w).astype(bool)]=j+1
                assert np.array_equal(z,lab)
                if gt is not None and q['metrics'][variant]['reference']['boundary']:
                    assert abs(boundary_f1(edge(lab),edge(gt),valid,2)-q['metrics'][variant]['reference']['boundary']['f1'])<1e-12;references+=1
            checks+=1
protocol=json.loads((R/'sam3_hf_protocol.json').read_text());assert protocol['user_supplied_prompts']==[] and protocol['parameters']['pred_iou_thresh']==.88 and protocol['parameters']['stability_score_thresh']==.95
out={'complete':bool(complete),'source_images':80,'distinct_source_checksums':80,'saved_partition_checks':checks,'independent_reference_f1_checks':references,'finished_treatments':sum(counts.values()),'expected_treatments':560,'counts':counts}
(R/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
