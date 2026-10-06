"""Freeze a balanced, rank-stratified pilot before observing new outcomes."""

from research_paths import research_path
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = research_path()
OUT = ROOT / 'outputs/moin-2025-analysis/multiscale-pilot'
manifest_path = ROOT / 'outputs/moin-2025-analysis/fourflip-benchmark/manifest.json'
rank_path = ROOT / 'outputs/moin-2025-analysis/plot-all-report/results.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
ranks = {r['id']: r for r in json.loads(rank_path.read_text(encoding='utf-8'))['images']}
rng = np.random.default_rng(20261001)
selected = []
for site in sorted({r['site'] for r in manifest['images']}):
    ordered = sorted([r for r in manifest['images'] if r['site'] == site],
                     key=lambda r: ranks[r['id']]['visual_rank'])
    for q, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), 1):
        position = int(rng.choice(indices))
        rec = dict(ordered[position])
        rec.update(selection_quintile=q, within_site_rank=position + 1,
                   prior_visual_rank=ranks[rec['id']]['visual_rank'],
                   prior_visual_score=ranks[rec['id']]['visual_score'])
        selected.append(rec)
assert len(selected) == len({r['source_sha256'] for r in selected}) == 20
data = {'version': 'multiscale-pilot-v1.0', 'seed': 20261001, 'unique_images': 20,
        'selection': 'Five photos per site: sort by existing visual rank, split into five nearly equal quintiles, draw one photo uniformly from each quintile using a fixed NumPy RNG. Quintile 1 has the highest prior visual diversity. No new benchmark outcomes used for selection.',
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_ranking_sha256': hashlib.sha256(rank_path.read_bytes()).hexdigest(),
        'images': selected}
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / 'selection.json'
if path.exists():
    assert json.loads(path.read_text(encoding='utf-8')) == data, 'Selection already frozen; do not overwrite a different cohort.'
else:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
print('Frozen 20-image selection; site/session blocks:', len({(r['site'],r['session']) for r in selected}))
for r in selected:
    print(r['id'], r['site'], r['session'], 'quintile', r['selection_quintile'], 'rank', r['prior_visual_rank'])
