"""One-image numerical check against the frozen 10% feature grid; not a new benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from vividRGB import FocusConfig, PCAProjection, extract_features
from vividRGB.encoders import DINOv3Encoder
from vividRGB.geometry import resize_bilinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image', default='moin_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    primary = args.root / 'runs/eco_collection75_v1' / args.image
    frozen = args.root / 'runs/eco_auto_clusters50_v1' / args.image
    source = json.loads((primary / 'source.json').read_text())
    assert hashlib.sha256((primary / 'input.png').read_bytes()).hexdigest() == source['input_sha256']
    encoder = DINOv3Encoder(device='cuda', local_files_only=True)
    encoder.warm_up()
    projection = PCAProjection.load(primary / 'transform.npz')
    with threadpool_limits(limits=2):
        result = extract_features(Image.open(primary / 'input.png'), encoder, projection, FocusConfig(focus=.10))
    grid, ys, xs = result.grid(256)
    with np.load(frozen / 's10_features.npz', allow_pickle=False) as z:
        expected = z['features']
        np.testing.assert_array_equal(ys, z['ys'])
        np.testing.assert_array_equal(xs, z['xs'])
    error = grid - expected
    np.testing.assert_allclose(grid, expected, atol=1e-4, rtol=1e-4)
    rng = np.random.default_rng(6)
    probe = rng.normal(size=(7, 11, 4)).astype(np.float32)
    np.testing.assert_allclose(resize_bilinear(probe, (19, 23), 'numpy'),
                               resize_bilinear(probe, (19, 23), 'torch'), atol=2e-6, rtol=2e-6)
    report = {'complete': True, 'image': args.image, 'method': 's10',
              'working_dimensions': source['working_dimensions'], 'grid_shape': list(grid.shape),
              'max_absolute_error': float(np.abs(error).max()), 'rms_error': float(np.sqrt(np.mean(error ** 2))),
              'relative_tolerance': 1e-4, 'absolute_tolerance': 1e-4,
              'source_input_sha256': source['input_sha256'],
              'torch': torch.__version__, 'encoder': encoder.metadata,
              'extraction': result.metadata,
              'scope': 'One frozen MOIN image; implementation equivalence, not ecological accuracy or a renewed benchmark'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

