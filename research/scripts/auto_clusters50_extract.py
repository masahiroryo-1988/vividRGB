"""Reuse ten cached grids; reconstruct the remaining forty with the frozen pipeline."""

from research_paths import research_path
import json, shutil
from pathlib import Path
from threadpoolctl import threadpool_limits
import clustering_methods10 as prior

ROOT=Path(str(research_path('runs/eco_auto_clusters50_v1')))

def prepare():
    ROOT.mkdir(exist_ok=True)
    rows=[]
    for ds in prior.DATASETS:
        selected=json.loads((prior.old.ROOT/f'{ds}_selection.json').read_text())['images']
        assert len(selected)==10
        rows.extend(selected)
    assert len({r['id'] for r in rows})==50
    prior.mb.save(ROOT/'selection.json',{'images':rows,'selection':'All ten previously collected examples per retained dataset; frozen before automatic cluster-count selection.'})
    for row in rows:
        d=ROOT/row['id'];d.mkdir(exist_ok=True)
        src=prior.ROOT/row['id']
        if src.exists():
            for name in ['input.jpg','source.json','transform.npz',*[f'{m}_{tail}' for m in prior.METHODS for tail in ['features.npz','extraction.json','pca.jpg']]]:
                if (src/name).exists() and not (d/name).exists():shutil.copy2(src/name,d/name)
    return rows

if __name__=='__main__':
    rows=prepare();prior.ROOT=ROOT
    with threadpool_limits(limits=2):
        prior.METHODS=['whole'];prior.extract(rows)
        prior.METHODS=['s05','s10','s20'];prior.extract(rows)
    prior.mb.save(ROOT/'extraction_complete.json',{'images':50,'feature_maps':200})
