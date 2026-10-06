"""Independent original-PCA rendering checks and exact cache-reuse checks."""
import json
import numpy as np
from PIL import Image
import auto_clusters50 as run

rows=json.loads((run.ROOT/'selection.json').read_text())['images'];checks=[];exact=0
for row in rows:
    d=run.ROOT/row['id'];old=run.extraction.prior.old.ROOT/row['id']
    assert run.digest(old/'input.png')==row['input_sha256']
    for method in run.METHODS:
        f=d/f'{method}_features.npz'
        if not f.exists() or not (d/f'{method}_extraction.json').exists():continue
        z=np.load(f);grid=z['features'];tr=np.load(d/'transform.npz')
        original=np.asarray(Image.open(old/f'{method}_pca.png'))[z['ys'][:,None],z['xs'][None,:]]
        rendered=np.uint8(np.clip((grid[:,:,:3]-tr['low'])/np.maximum(tr['high']-tr['low'],1e-6),0,1)*255)
        err=np.abs(original.astype(float)-rendered.astype(float))
        record={'id':row['id'],'method':method,'mean_rgb_error_255':float(err.mean()),'max_rgb_error_255':float(err.max()),'identical_rgb_fraction':float(np.all(original==rendered,axis=-1).mean())}
        assert err.mean()<3,record
        cached=run.extraction.prior.ROOT/row['id']/f'{method}_features.npz'
        if cached.exists():assert run.digest(f)==run.digest(cached);exact+=1
        checks.append(record)
result={'complete':len(checks)==200,'rendering_checks':len(checks),'exact_original_cache_checks':exact,'maximum_mean_rgb_error_255':max(c['mean_rgb_error_255'] for c in checks),'checks':checks,'scope':'Original independent full-resolution PCA PNG sampled at saved coordinates versus reconstruction from new grids. Only the three displayed PCs are checked by rendering; all16 cached coordinates are checked exactly for the earlier10-image subset. No biological accuracy claim.'}
run.save(run.ROOT/'feature_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
