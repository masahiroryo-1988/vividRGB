"""Recreate missing RGB overlays after obtaining the PMID photographs upstream.

Saved labels, fit records, source hashes and numerical results remain unchanged.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--images', nargs='*', help='Image IDs; default the ten PMID examples')
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ['VIVIDRGB_RESEARCH_ROOT'] = str(root)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research/scripts'))
    from pca_whitening10 import render
    images = args.images or [f'pmid_{i:02d}' for i in range(1, 11)]
    count = 0
    for image in images:
        source = root / 'runs/eco_auto_clusters50_v1' / image
        destination = root / 'runs/eco_gmm_final50_v1' / image
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['input.jpg', 'source.json', 'whole_pca.jpg', 's05_pca.jpg', 's10_pca.jpg', 's20_pca.jpg']:
            if not (destination / name).exists():
                shutil.copy2(source / name, destination / name)
        rgb = np.asarray(Image.open(source / 'input.jpg').convert('RGB'))
        for path in sorted(destination.glob('*_labels.npz')):
            stem = path.stem.removesuffix('_labels')
            if all((destination / (stem + tail)).exists()
                   for tail in ['_clusters.png', '_overlay.jpg', '_boundary.jpg']):
                continue
            method, pc, *_ = stem.split('_')
            with np.load(path, allow_pickle=False) as archive:
                labels = archive['labels']
            with np.load(source / f'{method}_{pc}_kmeans_labels.npz', allow_pickle=False) as archive:
                base = archive['labels']
            render(destination, stem, labels, base, rgb)
            count += 1
        print('Restored previews for', image)
    print('Rendered', count, 'partitions; frozen labels and fit JSON unchanged')


if __name__ == '__main__':
    main()
