"""Command-line workflow. Use frozen projections to compare methods on the same image."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import __version__
from .features import FocusConfig, calibrate, extract_features, open_rgb
from .projection import PCAProjection, varimax
from .clustering import cluster_kmeans, cluster_gmm
from .visualization import pca_rgb, labels_rgb


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vividrgb', description='Dense visual-cue discovery with forced focused attention')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('analyze', help='Extract features and save PCA/varimax/cluster maps')
    extract.add_argument('image', type=Path)
    extract.add_argument('--output', type=Path, required=True)
    extract.add_argument('--method', choices=['local', 'dino'], default='local')
    extract.add_argument('--focus', type=float, choices=[.05, .10, .20, .40], default=.10)
    extract.add_argument('--projection', type=Path, help='Reuse a shared calibration npz instead of fitting')
    extract.add_argument('--algorithm', choices=['kmeans', 'gmm'], default='kmeans')
    extract.add_argument('--k', default='auto', help='auto or integer (fixed 6 for the primary F1 benchmark)')
    extract.add_argument('--penalty', type=float, default=4)
    extract.add_argument('--device', default=None)
    extract.add_argument('--grid-side', type=int, default=256)
    extract.add_argument('--local-files-only', action='store_true')
    extract.add_argument('--model', default=None, help='Optional local checkpoint or model id')
    extract.add_argument('--revision', default=None)
    args = parser.parse_args(argv)
    from .encoders import DINOv3Encoder, DINO_MODEL, DINO_REVISION
    encoder = DINOv3Encoder(args.model or DINO_MODEL, args.revision or DINO_REVISION,
                            args.device, args.local_files_only)
    encoder.warm_up()
    im = open_rgb(args.image)
    projection = PCAProjection.load(args.projection) if args.projection else calibrate(im, encoder)
    result = extract_features(im, encoder, projection, FocusConfig(focus=args.focus), args.method)
    args.output.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    grid, ys, xs = result.grid(args.grid_side)
    np.savez_compressed(args.output / 'grid.npz', features=grid, ys=ys, xs=xs)
    rotated = varimax(projection.basis)
    rotation = rotated.pop('rotation')
    np.savez_compressed(args.output / 'varimax.npz', rotation=rotation)
    (args.output / 'varimax.json').write_text(json.dumps(rotated, indent=2), encoding='utf-8')
    limits = (projection.low, projection.high) if projection.low is not None else None
    Image.fromarray(pca_rgb(result.features, limits=limits)).save(args.output / 'pca.png')
    Image.fromarray(pca_rgb(result.features, rotation)).save(args.output / 'pca_varimax.png')
    if args.algorithm == 'kmeans':
        k = 'auto' if args.k == 'auto' else int(args.k)
        clusters = cluster_kmeans(grid, k=k)
    else:
        clusters = cluster_gmm(grid, penalty=args.penalty)
    np.savez_compressed(args.output / 'clusters.npz', labels=clusters.labels)
    Image.fromarray(labels_rgb(clusters.labels)).resize(im.size, Image.Resampling.NEAREST).save(args.output / 'clusters.png')
    (args.output / 'clustering.json').write_text(json.dumps(clusters.metadata, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'selected_k': clusters.metadata['selected_k'],
                      'extraction_seconds': result.metadata['extraction_seconds']}))


if __name__ == '__main__':
    main()

