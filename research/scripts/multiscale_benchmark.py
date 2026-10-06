"""Matched, four-flip multiscale pilot using the frozen Phase 1 evaluator."""

from research_paths import research_path
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_benchmark as p1
from phase1_fourflip_benchmark import fourflip_encode, _predict, _dense

VERSION = 'multiscale-pilot-v1.0'
SCALES = [5, 10, 20, 40]
TREATMENTS = {'s05': [5], 's10': [10], 's20': [20], 's40': [40],
              's05_10': [5, 10], 's10_20': [10, 20],
              's05_10_20': [5, 10, 20], 's10_20_40': [10, 20, 40],
              's05_10_20_40': [5, 10, 20, 40]}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def geometry(w, h, scale):
    crop = max(16, round(min(w, h) * scale / 100))
    stride = max(1, round(crop * 2 / 3))
    boxes = [(x,y,x+crop,y+crop) for y in p1.starts(h,crop,stride) for x in p1.starts(w,crop,stride)]
    return crop, stride, boxes


def project(z, basis, mu):
    flat = z.reshape(-1, 768)
    pr = np.empty((len(flat), 16), np.float32)
    for st in range(0, len(flat), 8192):
        pr[st:st+8192] = (flat[st:st+8192].astype(np.float32)-mu) @ basis
    return pr.reshape(*z.shape[:3], 16)


def add_tiles(field, weights, projected, boxes, blend, torch, F):
    for pr, (x0,y0,x1,y1) in zip(projected, boxes):
        d = F.interpolate(torch.from_numpy(pr).permute(2,0,1)[None],
                          size=(y1-y0,x1-x0), mode='bilinear', align_corners=False)[0].permute(1,2,0).numpy()
        field[y0:y1,x0:x1] += d * blend
        weights[y0:y1,x0:x1] += blend


def extract(im, scale, basis, mu, encode, torch, F):
    w,h = im.size
    crop,stride,boxes = geometry(w,h,scale)
    field = np.zeros((h,w,16), np.float32)
    weight = np.zeros((h,w,1), np.float32)
    blend = (np.maximum(np.hanning(crop),.05)[:,None]*np.maximum(np.hanning(crop),.05)[None,:])[...,None].astype(np.float32)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    encoding_seconds = norm_sum = norm_sq_sum = 0.0
    token_count = 0
    for st in range(0,len(boxes),12):
        chunk = boxes[st:st+12]
        t = time.perf_counter()
        crops = [im.crop(b).resize((384,384), Image.Resampling.BICUBIC) for b in chunk]
        z = fourflip_encode(crops, encode)
        torch.cuda.synchronize()
        encoding_seconds += time.perf_counter()-t
        z32 = z.astype(np.float32)
        norms = np.linalg.norm(z32,axis=-1)
        norm_sum += float(norms.sum(dtype=np.float64))
        norm_sq_sum += float((norms**2).sum(dtype=np.float64))
        token_count += norms.size
        pr = project(z, basis, mu)
        add_tiles(field,weight,pr,chunk,blend,torch,F)
    assert np.all(weight>0), 'Uncovered output pixels'
    field /= weight
    meta = {'scale_percent':scale, 'crop_px':crop, 'stride_px':stride, 'tiles':len(boxes),
            'input_size':384, 'token_count_per_orientation':token_count, 'orientations':4,
            'encoding_seconds':encoding_seconds, 'extraction_seconds':time.perf_counter()-start,
            'mean_raw_fourflip_token_norm':norm_sum/token_count,
            'rms_raw_fourflip_token_norm':math.sqrt(norm_sq_sum/token_count),
            'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,
            'minimum_blend_weight':float(weight.min())}
    return field,meta


def protocol(selection):
    return {'version':VERSION, 'selection_sha256':p1.sha(selection),
        'code_sha256':p1.sha(Path(__file__)), 'phase1_evaluator_sha256':p1.sha(Path(p1.__file__)),
        'fourflip_evaluator_sha256':p1.sha(Path(__file__).with_name('phase1_fourflip_benchmark.py')),
        'model':p1.MODEL, 'model_revision':p1.REV, 'scales_percent_short_side':SCALES,
        'treatments':TREATMENTS, 'baseline':'s10', 'primary_k':6, 'sensitivity_k':[4,8],
        'input':'Existing cleaned working RGB, longest side <=1280. Crop size is round(shorter side * fraction), minimum16. Crop stride round(2/3 crop). Bicubic resize each crop to384x384. Processor RGB /255 and mean/std.',
        'features':'Raw 768D DINOv3 tokens, no explicit L2 normalization. Four orientations (identity, horizontal, vertical, both), inverse-reflect token grids, average in float32, cast float16 as in prior benchmark.',
        'fusion':'Project all scales with the same frozen per-photo centered PCA16. Bilinear interpolation, Hann blend (per-axis floor0.05), normalize overlap weights separately for each scale. Arithmetic mean of these aligned scale fields, equal weights summing to1. Centered affine projection and unit-sum averaging commute, so fusion before or after this fixed projection is equivalent up to floating point. PCA colors and cluster IDs are never averaged.',
        'clustering':'Separate KMeans per treatment; K=4,6,8, n_init10, seed42; same4096 frozen eligible coordinates. Frozen PCA/display limits/valid mask; no shared cluster centers.',
        'mask':'Prior artifact exclusions plus16px erosion; whole-image vegetation content is otherwise retained. No vegetation scoring mask.',
        'primary_score':'RGB-edge boundary F1, K6. Grayscale sigma1 Sobel, strongest15% eligible gradients,2px tolerance. Appearance proxy; not leaf IoU or botanical accuracy.',
        'seams':'Existing contrast/orientation-matched seam ratio. All treatments evaluated against every one of the four identical test grids; 10% grid primary, other grids sensitivity. Distance |ratio-1| is descriptive and may miss visible lattice artifacts.',
        'runtime':'One GPU worker,12 crops/batch. Per-scale encoding time includes RGB crop preparation and four model orientations, excludes projection/blending/clustering. Mixture encoding cost is the sum of constituent scale times; reuse means it is an estimated standalone encoding cost, not an independently timed full run.',
        'cache':'Float32 PCA16 scale fields and K6 labels retained on server for reuse. Results and previews copied locally.',
        'uncertainty':'Exploratory pilot; balanced sites and rank strata. Paired site/session-block bootstrap stratified by site, equal site weights,5000 draws. Repeated spatial plots and rank-selection uncertainty are not fully modeled. No multiplicity adjustment; not confirmatory inference.',
        'leaf_accuracy':'Not measured: no reviewed leaf-instance masks.'}


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
    base,phase1,prior,out = map(Path,[args.base,args.phase1,args.prior,args.out])
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    prot = protocol(selection_path)
    phash = digest(prot)
    if (out/'protocol.json').exists():
        assert digest(json.loads((out/'protocol.json').read_text())) == phash, 'Protocol changed; choose a new run directory'
    save(out/'protocol.json',prot)
    save(out/'selection.json',selection)
    rows = selection['images']
    old = {r['id']:r for r in json.loads((prior/'results.json').read_text())['images']}
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    proc=AutoImageProcessor.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True)
    model=AutoModel.from_pretrained(p1.MODEL,revision=p1.REV,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    mean=torch.tensor(proc.image_mean,device='cuda')[None,:,None,None]
    std=torch.tensor(proc.image_std,device='cuda')[None,:,None,None]
    def encode(images):
        arr=np.stack([np.asarray(im) for im in images])
        pix=torch.from_numpy(arr).permute(0,3,1,2).cuda().float()/255
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            z=model(pixel_values=(pix-mean)/std).last_hidden_state[:,1+model.config.num_register_tokens:]
        return z.reshape(len(images),24,24,768).half().cpu().numpy()
    encode([Image.new('RGB',(384,384))])
    completed=0
    for rec in rows:
        key=rec['id']; dest=out/key
        if (dest/'metrics.json').exists():
            assert json.loads((dest/'metrics.json').read_text())['protocol_hash']==phash
            continue
        if args.limit and completed>=args.limit: break
        start=time.perf_counter()
        try:
            dest.mkdir(parents=True,exist_ok=True)
            image_path=base/'clean_inputs'/f'{key}.png'
            assert p1.sha(image_path)==rec['clean_input_sha256']
            im=Image.open(image_path).convert('RGB'); rgb=np.asarray(im); w,h=im.size
            valid=binary_erosion(~np.load(base/key/'artifact_exclusion.npz')['excluded'],iterations=16,border_value=0)
            frozen_path=phase1/key/'transform.npz'; frozen=np.load(frozen_path)
            basis,mu=frozen['pca_basis'],frozen['pca_mean']; coords=frozen['training_flat_coordinates']
            assert valid.ravel()[coords].all()
            thumb=im.copy(); thumb.thumbnail((640,640)); thumb.save(dest/'original.jpg',quality=90)
            fields={}; scales={}
            with threadpool_limits(limits=2):
                for s in SCALES:
                    field_path=dest/f'field_s{s:02d}.npy'; meta_path=dest/f'field_s{s:02d}.json'
                    if field_path.exists() and meta_path.exists():
                        meta=json.loads(meta_path.read_text()); assert meta['protocol_hash']==phash
                        field=np.load(field_path)
                    else:
                        field,meta=extract(im,s,basis,mu,encode,torch,F)
                        meta['protocol_hash']=phash
                        np.save(field_path,field)
                        save(meta_path,meta)
                    assert field.shape==(h,w,16) and np.isfinite(field).all()
                    meta['projected_spatial_sd']=float(np.sqrt(np.var(field.reshape(-1,16)[coords],axis=0).mean()))
                    fields[s]=field; scales[str(s)]=meta
                    print('SCALE',key,s,'tiles',meta['tiles'],'seconds',round(meta['extraction_seconds'],1),flush=True)
            edge,_=p1.rgb_edges(rgb,valid)
            methods={}; label_store={'valid':valid}; checks={}
            with threadpool_limits(limits=2):
                for name,ss in TREATMENTS.items():
                    field=fields[ss[0]] if len(ss)==1 else np.mean(np.stack([fields[s] for s in ss]),axis=0,dtype=np.float32)
                    row={'scales':ss,'scale_weights':[1/len(ss)]*len(ss),'k':{},'seam_grids':{},
                         'estimated_encoding_seconds':sum(scales[str(s)]['encoding_seconds'] for s in ss),
                         'crop_inferences':4*sum(scales[str(s)]['tiles'] for s in ss)}
                    for grid in SCALES:
                        c,st,_=geometry(w,h,grid)
                        row['seam_grids'][str(grid)]=p1.seam_score(field,rgb,valid,c,st)
                    for k in (4,6,8):
                        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(field.reshape(-1,16)[coords])
                        labels=_predict(km,field,w,h)
                        areas=np.bincount(labels[valid],minlength=k)/valid.sum()
                        row['k'][str(k)]={'rgb_edge':p1.bf1(p1.boundaries(labels),edge,valid),
                            'boundary_density':float(p1.boundaries(labels)[valid].mean()),
                            'effective_clusters':float(np.exp(-np.sum(areas[areas>0]*np.log(areas[areas>0])))),
                            'occupied_clusters':int((areas>0).sum()),'dominant_fraction':float(areas.max())}
                        if k==6:
                            label_store[name]=labels.astype(np.uint8)
                            overlay=np.uint8(.55*rgb+.45*p1.PALETTE[labels]); overlay[~valid]=145
                            thumb=Image.fromarray(overlay); thumb.thumbnail((640,640)); thumb.save(dest/f'{name}_overlay.jpg',quality=90)
                            boundary_rgb=rgb.copy(); boundary_rgb[p1.boundaries(labels)&valid]=[255,40,160]; boundary_rgb[~valid]=145
                            thumb=Image.fromarray(boundary_rgb); thumb.thumbnail((640,640)); thumb.save(dest/f'{name}_boundary.jpg',quality=90)
                    color=np.uint8(np.clip((field[:,:,:3]-frozen['rgb_low'])/np.maximum(frozen['rgb_high']-frozen['rgb_low'],1e-6),0,1)*255)
                    color[~valid]=145
                    thumb=Image.fromarray(color); thumb.thumbnail((640,640)); thumb.save(dest/f'{name}_pca.jpg',quality=90)
                    methods[name]=row
            old_labels=np.load(prior/key/'labels.npz')['crop384_k6']
            checks['baseline_label_ari']=float(adjusted_rand_score(old_labels[valid],label_store['s10'][valid]))
            checks['baseline_f1_delta']={str(k):methods['s10']['k'][str(k)]['rgb_edge']['f1']-old[key]['methods']['crop384_fourflip']['k'][str(k)]['rgb_edge']['f1'] for k in (4,6,8)}
            checks['baseline_seam_delta']=methods['s10']['seam_grids']['10']['ratio']-old[key]['methods']['crop384_fourflip']['seam']['ratio']
            checks['baseline_reproduction_passed']=checks['baseline_label_ari']>.99 and max(abs(x) for x in checks['baseline_f1_delta'].values())<.002
            if not checks['baseline_reproduction_passed']:
                raise ValueError('Baseline reproduction failed: '+str(checks))
            np.savez_compressed(dest/'labels.npz',**label_store)
            result={**rec,'version':VERSION,'protocol_hash':phash,'pca_transform_sha256':p1.sha(frozen_path),
                    'eligible_pixels':int(valid.sum()),'methods':methods,'scales':scales,'checks':checks,
                    'total_seconds':time.perf_counter()-start}
            save(dest/'metrics.json',result)
            completed+=1
            n=len(list(out.glob('*/metrics.json')))
            save(out/'progress.json',{'completed':n,'total':len(rows),'last_id':key,'last_seconds':result['total_seconds'],'state':'complete' if n==len(rows) else 'running'})
            print('DONE',key,round(result['total_seconds'],1),'baseline',checks,flush=True)
            del fields,field,label_store
            torch.cuda.empty_cache()
        except Exception:
            save(out/'failure.json',{'id':key,'traceback':traceback.format_exc()})
            raise


def selftest():
    import torch
    import torch.nn.functional as F
    rng=np.random.default_rng(2)
    mu=rng.normal(size=768).astype(np.float32)
    basis=rng.normal(size=(768,16)).astype(np.float32)
    raw=rng.normal(size=(4,7,768)).astype(np.float32)
    np.testing.assert_allclose((raw.mean(0)-mu)@basis,((raw-mu)@basis).mean(0),rtol=1e-4,atol=5e-5)
    # Batch streaming must agree with the original 10% reconstruction, including overlap normalization.
    w,h,crop,stride=91,73,16,11
    boxes=[(x,y,x+crop,y+crop) for y in p1.starts(h,crop,stride) for x in p1.starts(w,crop,stride)]
    z=rng.normal(size=(len(boxes),24,24,768)).astype(np.float16)
    with threadpool_limits(limits=2):
        reference=_dense(z,basis,mu,'crop384_fourflip',boxes,w,h,crop,torch,F)
        field=np.zeros((h,w,16),np.float32); weights=np.zeros((h,w,1),np.float32)
        blend=(np.maximum(np.hanning(crop),.05)[:,None]*np.maximum(np.hanning(crop),.05)[None,:])[...,None].astype(np.float32)
        for st in range(0,len(boxes),12): add_tiles(field,weights,project(z[st:st+12],basis,mu),boxes[st:st+12],blend,torch,F)
        assert weights.min()>0
        np.testing.assert_allclose(field/weights,reference,rtol=1e-4,atol=1e-4)
    # Equal-scale averaging is independent of per-scale tile multiplicity.
    a=np.full((3,4,16),2.);b=np.full_like(a,8.)
    assert np.all(np.mean([a,b],axis=0)==5.)
    print('PASS: affine fusion, streaming reconstruction, full spatial coverage, equal-scale weighting')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['run','selftest'])
    ap.add_argument('--base',default=str(research_path('runs/moin_plot_all_20260928')))
    ap.add_argument('--phase1',default=str(research_path('runs/moin_phase1_benchmark_v1')))
    ap.add_argument('--prior',default=str(research_path('runs/moin_phase1_fourflip_v1')))
    ap.add_argument('--out',default=str(research_path('runs/moin_multiscale_pilot_v1')))
    ap.add_argument('--selection',default=str(research_path('work/multiscale_selection.json')))
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.command=='selftest': selftest()
    else: run(args)
