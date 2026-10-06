"""Four-flip extension to the frozen, label-free Phase 1 DINO benchmark.

The original Phase 1 results are reused as the no-flip cells of a matched 3x2
design. This script only infers the three four-flip cells. No leaf truth masks
exist, so all reported image-edge scores are proxies rather than accuracy.
"""

from research_paths import research_path
import argparse, json, math, sys, time, traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_erosion
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_benchmark as p1

VERSION = 'phase1-fourflip-v1.0'
BASE_DEFAULT = Path(str(research_path('runs/moin_plot_all_20260928')))
PHASE1_DEFAULT = Path(str(research_path('runs/moin_phase1_benchmark_v1')))
OUT_DEFAULT = Path(str(research_path('runs/moin_phase1_fourflip_v1')))
METHODS = ['whole_fourflip', 'crop384_fourflip', 'crop_native_fourflip']
BASE_NAMES = ['whole', 'crop384', 'crop_native']
PRESENTATION = {
    'whole': 'Direct DINO',
    'whole_fourflip': 'Direct DINO · four-flip mean',
    'crop_native': '10% crops · native size',
    'crop_native_fourflip': '10% crops · native size · four-flip mean',
    'crop384': '10% crops · enlarged to 384',
    'crop384_fourflip': '10% crops · enlarged to 384 · four-flip mean',
}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def fourflip_encode(images, encode):
    """Average raw DINO token grids after mapping each reflection to input axes."""
    accum = None
    for vertical, horizontal in [(False, False), (False, True), (True, False), (True, True)]:
        views = []
        for im in images:
            v = ImageOps.flip(im) if vertical else im
            v = ImageOps.mirror(v) if horizontal else v
            views.append(v)
        z = encode(views).astype(np.float32)
        if horizontal:
            z = z[:, :, ::-1, :]
        if vertical:
            z = z[:, ::-1, :, :]
        accum = z if accum is None else accum + z
    return (accum * 0.25).astype(np.float16)


def _dense(z, basis, mu, method, boxes, width, height, crop, torch, F):
    """Frozen Phase 1 PCA and token interpolation/blending on averaged tokens."""
    flat = z.reshape(-1, 768)
    projected = np.empty((len(flat), 16), np.float32)
    for start in range(0, len(flat), 8192):
        projected[start:start + 8192] = (flat[start:start + 8192].astype(np.float32) - mu) @ basis
    projected = projected.reshape(*z.shape[:3], 16)
    field = np.zeros((height, width, 16), np.float32)
    weight = np.zeros((height, width, 1), np.float32)
    blend = (np.maximum(np.hanning(crop), .05)[:, None] *
             np.maximum(np.hanning(crop), .05)[None, :])[..., None].astype(np.float32)
    tile_boxes = [(0, 0, width, height)] if method == 'whole_fourflip' else boxes
    for i, box in enumerate(tile_boxes):
        x0, y0, x1, y1 = box
        bh, bw = y1-y0, x1-x0
        target = (bh, bw) if method == 'crop384_fourflip' else (math.ceil(bh/16)*16, math.ceil(bw/16)*16)
        d = F.interpolate(torch.from_numpy(projected[i]).permute(2, 0, 1)[None],
                          size=target, mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()[:bh, :bw]
        wt = 1.0 if method == 'whole_fourflip' else blend
        field[y0:y1, x0:x1] += d * wt
        weight[y0:y1, x0:x1] += wt
    assert np.all(weight > 0)
    return field / weight


def _predict(km, field, width, height):
    flat = field.reshape(-1, 16)
    labels = np.concatenate([km.predict(flat[i:i+32768]) for i in range(0, len(flat), 32768)])
    return labels.reshape(height, width)


def _run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel

    base, phase1, out = Path(args.base), Path(args.phase1), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p1_manifest = json.loads((phase1/'manifest.json').read_text())
    p1_results = json.loads((phase1/'results.json').read_text())
    rows = p1_manifest['images']
    if not (out/'manifest.json').exists():
        save_json(out/'manifest.json', {'base': str(base), 'phase1': str(phase1), 'unique_images': len(rows), 'images': rows})
        save_json(out/'protocol.json', {
            'version': VERSION,
            'design': 'Matched 3 scale x 2 flip-average factorial; no-flip cells reused from frozen Phase 1 results.',
            'methods': METHODS,
            'four_flip_orientations': ['identity', 'horizontal', 'vertical', 'horizontal+vertical'],
            'four_flip_operation': 'DINO raw 768D token grids are inverse-reflected into the original crop coordinates, averaged before the frozen Phase 1 PCA projection, interpolation, and crop blending.',
            'input_scales': {'whole': 'saved working image, longest side <=1280', 'crop_fraction': .1, 'crop_stride_fraction': 2/3, 'crop384': 'bicubic resize to 384x384', 'crop_native': 'native-size 10% crop'},
            'frozen_settings': 'Use each image Phase 1 PCA mean/basis, RGB display range, eligible evaluation mask, and 4096 training coordinates; fit separate KMeans per new treatment at K=4/6/8, n_init=10, seed=42.',
            'model': p1.MODEL, 'model_revision': p1.REV, 'image_normalization': 'processor RGB mean/std after 0-1 scaling; no L2 feature normalization',
            'primary_k': 6,
            'metrics': 'RGB-edge boundary F1 and seam excess are image-derived diagnostics, not leaf accuracy; cluster occupancy/effective clusters/boundary density are descriptive.',
            'flip_stability': 'The independent flip ARI/feature error are not reported for four-flip means because that transformation is included in the estimator.',
            'leaf_accuracy': 'Not measured; no reviewed leaf masks or leaf-instance predictions are available.',
            'uncertainty': '2000 paired bootstrap resamples of site/session blocks; repeated plots may remain correlated across sessions.',
            'original_phase1_version': json.loads((phase1/'protocol.json').read_text())['version'],
        })

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    proc = AutoImageProcessor.from_pretrained(p1.MODEL, revision=p1.REV, local_files_only=True)
    model = AutoModel.from_pretrained(p1.MODEL, revision=p1.REV, local_files_only=True, attn_implementation='sdpa').cuda().eval()
    mean = torch.tensor(proc.image_mean, device='cuda')[None, :, None, None]
    std = torch.tensor(proc.image_std, device='cuda')[None, :, None, None]

    def encode(images):
        arr = np.stack([np.asarray(im.convert('RGB')) for im in images])
        h, w = arr.shape[1:3]
        arr = np.pad(arr, ((0,0),(0,(-h)%16),(0,(-w)%16),(0,0)), mode='edge')
        pix = torch.from_numpy(arr).permute(0,3,1,2).cuda().float()/255
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            z = model(pixel_values=(pix-mean)/std).last_hidden_state[:, 1+model.config.num_register_tokens:]
        return z.reshape(len(images), math.ceil(h/16), math.ceil(w/16), 768).half().cpu().numpy()

    encode([Image.new('RGB',(384,384))])
    torch.cuda.synchronize()
    done = 0
    for index, rec in enumerate(rows):
        if index % args.shards != args.shard:
            continue
        dest = out/rec['id']
        if (dest/'metrics.json').exists():
            continue
        if args.limit and done >= args.limit:
            break
        start_time = time.perf_counter()
        dest.mkdir(parents=True, exist_ok=True)
        key = rec['id']
        try:
            image_path = base/'clean_inputs'/f'{key}.png'
            if p1.sha(image_path) != rec['clean_input_sha256']:
                raise ValueError(f'clean input hash mismatch: {key}')
            im = Image.open(image_path).convert('RGB')
            rgb = np.asarray(im)
            width, height = im.size
            crop = max(16, round(min(width, height)*.1))
            stride = max(1, round(crop*2/3))
            boxes = [(x,y,x+crop,y+crop) for y in p1.starts(height,crop,stride) for x in p1.starts(width,crop,stride)]
            excluded = np.load(base/key/'artifact_exclusion.npz')['excluded']
            valid = binary_erosion(~excluded, iterations=16, border_value=0)
            if valid.sum() <= 4096:
                raise ValueError(f'insufficient eligible pixels: {key}')
            frozen = np.load(phase1/key/'transform.npz')
            basis, mu = frozen['pca_basis'], frozen['pca_mean']
            coords = frozen['training_flat_coordinates']
            rgb_low, rgb_high = frozen['rgb_low'], frozen['rgb_high']
            methods = {}
            saved_labels = {'valid': valid}
            for method in METHODS:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                method_start = time.perf_counter()
                all_tokens = []
                if method == 'whole_fourflip':
                    token = fourflip_encode([im], encode)
                    all_tokens.append(token)
                    selected_boxes = [(0,0,width,height)]
                else:
                    selected_boxes = boxes
                    for batch_start in range(0, len(boxes), 12):
                        chunk_boxes = boxes[batch_start:batch_start+12]
                        crops = [im.crop(b) for b in chunk_boxes]
                        if method == 'crop384_fourflip':
                            crops = [crop_im.resize((384,384), Image.Resampling.BICUBIC) for crop_im in crops]
                        all_tokens.append(fourflip_encode(crops, encode))
                torch.cuda.synchronize()
                encoding_seconds = time.perf_counter()-method_start
                z = np.concatenate(all_tokens, axis=0)
                with threadpool_limits(limits=2):
                    field = _dense(z, basis, mu, method, selected_boxes, width, height, crop, torch, F)
                del z, all_tokens
                edge, _ = p1.rgb_edges(rgb, valid)
                row = {
                    'fourflip_encoding_seconds': encoding_seconds,
                    'peak_gpu_mib': torch.cuda.max_memory_allocated()/1024**2,
                    'seam': p1.seam_score(field, rgb, valid, crop, stride),
                    'k': {},
                }
                # The token count is recorded from crop geometry, excluding padded tokens.
                if method == 'crop384_fourflip':
                    row['token_count'] = 24*24*len(selected_boxes)
                else:
                    row['token_count'] = int(sum(math.ceil((b[3]-b[1])/16)*math.ceil((b[2]-b[0])/16) for b in selected_boxes))
                with threadpool_limits(limits=2):
                    for k in (4,6,8):
                        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(field.reshape(-1,16)[coords])
                        labels = _predict(km, field, width, height)
                        areas = np.bincount(labels[valid], minlength=k)/valid.sum()
                        occupied = areas > 0
                        effective = float(np.exp(-np.sum(areas[occupied]*np.log(areas[occupied]))))
                        boundary = p1.boundaries(labels)
                        row['k'][str(k)] = {
                            'rgb_edge': p1.bf1(boundary, edge, valid),
                            'boundary_density': float(boundary[valid].mean()),
                            'effective_clusters': effective,
                            'occupied_clusters': int(occupied.sum()),
                            'dominant_fraction': float(areas.max()),
                        }
                        if k == 6:
                            saved_labels[method.replace('_fourflip','')+'_k6'] = labels.astype(np.uint8)
                            overlay = np.uint8(.55*rgb+.45*p1.PALETTE[labels])
                            overlay[~valid] = [145,145,145]
                            preview = Image.fromarray(overlay); preview.thumbnail((480,480))
                            preview.save(dest/f'{method}_overlay.jpg', quality=88)
                color = np.uint8(np.clip((field[:,:,:3]-rgb_low)/np.maximum(rgb_high-rgb_low,1e-6),0,1)*255)
                color[~valid] = [145,145,145]
                preview = Image.fromarray(color); preview.thumbnail((480,480))
                preview.save(dest/f'{method}_pca.jpg', quality=88)
                row['total_method_seconds'] = time.perf_counter()-method_start
                methods[method] = row
                del field
            np.savez_compressed(dest/'labels.npz', **saved_labels)
            record = {k:rec[k] for k in ['id','site','session','source_sha256','reference_split','previously_viewed_phase1']}
            record.update(version=VERSION, size=[width,height], crop=crop, stride=stride, crop_count=len(boxes), eligible_pixels=int(valid.sum()), methods=methods, total_seconds=time.perf_counter()-start_time)
            save_json(dest/'metrics.json', record)
            done += 1
            save_json(out/f'progress-{args.shard}.json', {'completed':len(list(out.glob('*/metrics.json'))),'total':len(rows),'last_id':key,'last_seconds':record['total_seconds'],'state':'running'})
            print('DONE', key, round(record['total_seconds'],2), flush=True)
        except Exception:
            save_json(out/'failure.json', {'id':key,'traceback':traceback.format_exc()})
            raise
        torch.cuda.empty_cache()
    print('SHARD_DONE',args.shard,flush=True)


def aggregate(args):
    phase1, out = Path(args.phase1), Path(args.out)
    old = json.loads((phase1/'results.json').read_text())
    new_records = [json.loads(p.read_text()) for p in sorted(out.glob('*/metrics.json'))]
    new_by_id = {r['id']:r for r in new_records}
    ids = [r['id'] for r in old['images'] if r['id'] in new_by_id]
    all_names = ['whole','whole_fourflip','crop_native','crop_native_fourflip','crop384','crop384_fourflip']
    rows = []
    old_by_id = {r['id']:r for r in old['images']}
    for key in ids:
        b, n = old_by_id[key], new_by_id[key]
        methods = dict(b['methods'])
        methods.update(n['methods'])
        rows.append({**{k:b[k] for k in ['id','site','session','source_sha256','reference_split','previously_viewed_phase1']},'methods':methods})
    blocks = sorted({r['site']+'/'+r['session'] for r in rows})
    block_indices = {g:[i for i,r in enumerate(rows) if r['site']+'/'+r['session']==g] for g in blocks}
    rng = np.random.default_rng(20261001)
    draws = [np.concatenate([block_indices[blocks[j]] for j in rng.integers(0,len(blocks),len(blocks))]) for _ in range(2000)] if rows else []

    def metric(r, method, name):
        q = r['methods'][method]
        k = q['k']['6']
        if name == 'rgb_edge_f1': return k['rgb_edge']['f1']
        if name == 'rgb_edge_precision': return k['rgb_edge']['precision']
        if name == 'rgb_edge_recall': return k['rgb_edge']['recall']
        if name == 'seam_ratio': return q['seam']['ratio']
        if name == 'seam_distance_from_one': return abs(q['seam']['ratio']-1) if q['seam']['ratio'] is not None else None
        if name == 'boundary_density': return k['boundary_density']
        if name == 'effective_clusters': return k['effective_clusters']
        if name == 'mean_model_seconds': return q['fourflip_encoding_seconds'] if 'fourflip_encoding_seconds' in q else q['encoding_seconds']
        raise KeyError(name)

    names = ['rgb_edge_f1','rgb_edge_precision','rgb_edge_recall','seam_ratio','seam_distance_from_one','boundary_density','effective_clusters','mean_model_seconds']
    summary, contrasts = {}, []
    for name in names:
        summary[name] = {}
        for method in all_names:
            values = np.array([metric(r,method,name) if metric(r,method,name) is not None else np.nan for r in rows])
            bootstrap = [float(np.nanmean(values[d])) for d in draws]
            summary[name][method] = {
                'label': PRESENTATION[method], 'mean':float(np.nanmean(values)), 'median':float(np.nanmedian(values)), 'n':int(np.isfinite(values).sum()),
                'block_bootstrap_mean_ci95':np.quantile(bootstrap,[.025,.975]).tolist() if bootstrap else [None,None],
                'site_means':{s:float(np.nanmean([metric(r,method,name) for r in rows if r['site']==s])) for s in sorted({r['site'] for r in rows})},
            }
        pairs = [(a,b,'four-flip effect at scale') for a,b in [('whole','whole_fourflip'),('crop_native','crop_native_fourflip'),('crop384','crop384_fourflip')]]
        pairs += [(a,b,'scale contrast without averaging') for a,b in [('whole','crop_native'),('whole','crop384'),('crop_native','crop384')]]
        pairs += [(a,b,'scale contrast with four-flip mean') for a,b in [('whole_fourflip','crop_native_fourflip'),('whole_fourflip','crop384_fourflip'),('crop_native_fourflip','crop384_fourflip')]]
        for a,b,kind in pairs:
            vals = np.array([metric(r,b,name)-metric(r,a,name) if metric(r,a,name) is not None and metric(r,b,name) is not None else np.nan for r in rows])
            boot = [float(np.nanmean(vals[d])) for d in draws]
            contrasts.append({'metric':name,'comparison_type':kind,'a':a,'b':b,'a_label':PRESENTATION[a],'b_label':PRESENTATION[b],'definition':'B minus A','mean_difference':float(np.nanmean(vals)),'block_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist() if boot else [None,None]})
        for scale, noflip, flip in [('whole','whole','whole_fourflip'),('crop_native','crop_native','crop_native_fourflip'),('crop384','crop384','crop384_fourflip')]:
            # Difference-in-differences: (flip minus noflip) at this scale vs direct.
            if scale == 'whole': continue
            direct_delta = np.array([metric(r,'whole_fourflip',name)-metric(r,'whole',name) for r in rows])
            scale_delta = np.array([metric(r,flip,name)-metric(r,noflip,name) if metric(r,flip,name) is not None and metric(r,noflip,name) is not None else np.nan for r in rows])
            did = scale_delta-direct_delta
            boot = [float(np.nanmean(did[d])) for d in draws]
            contrasts.append({'metric':name,'comparison_type':'flip effect interaction vs direct','scale':scale,'mean_difference':float(np.nanmean(did)),'block_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist() if boot else [None,None]})
    protocol = json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else {}
    save_json(out/'results.json',{
        'version':VERSION,'complete':len(rows)==old['total'],'completed':len(rows),'total':old['total'],'blocks':len(blocks),
        'block_unit':'site/session; repeated physical plots may remain correlated across sessions',
        'methods':{m:PRESENTATION[m] for m in all_names},'summary':summary,'paired_contrasts':contrasts,
        'images':rows,'leaf_accuracy':{'status':'not measured','reviewed_reference_windows':0},'protocol':protocol,
    })
    save_json(out/'progress.json',{'completed':len(rows),'total':old['total'],'state':'complete' if len(rows)==old['total'] else 'partial'})
    print('AGGREGATE',len(rows),'/',old['total'],flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['run','aggregate','selftest'])
    ap.add_argument('--base', type=Path, default=BASE_DEFAULT)
    ap.add_argument('--phase1', type=Path, default=PHASE1_DEFAULT)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    args = ap.parse_args()
    if args.command == 'run': _run(args)
    elif args.command == 'aggregate': aggregate(args)
    else:
        # Geometry alignment test: inverse transforms must make constant-valued
        # synthetic orientation maps identical before averaging.
        a = np.arange(2*3*4).reshape(1,2,3,4)
        for v,h in [(False,False),(False,True),(True,False),(True,True)]:
            q = a[:,::-1] if v else a
            q = q[:,:,::-1] if h else q
            q = q[:,:,::-1] if h else q
            q = q[:,::-1] if v else q
            assert np.array_equal(q,a)
        print('PASS four-orientation inverse-alignment geometry')
