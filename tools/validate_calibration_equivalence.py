"""Validate balanced PCA calibration against one frozen image projection."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from vividRGB import PCAProjection, calibrate
from vividRGB.encoders import DINOv3Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image', default='moin_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    primary = args.root / 'runs/eco_collection75_v1' / args.image
    encoder = DINOv3Encoder(device='cuda', local_files_only=True)
    encoder.warm_up()
    with threadpool_limits(limits=2):
        projection = calibrate(primary / 'input.png', encoder)
    expected = PCAProjection.load(primary / 'transform.npz')
    errors = {}
    for name in ['basis', 'mean', 'low', 'high']:
        actual, reference = getattr(projection, name), getattr(expected, name)
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
        errors[name] = float(np.max(np.abs(actual - reference)))
    report = {'complete': True, 'image': args.image, 'max_absolute_errors': errors,
              'retained_components': 16, 'tokens_per_treatment': 2048,
              'sampling_seed': 44, 'pca_seed': 42,
              'scope': 'One-image balanced PCA calibration equivalence; not a renewed benchmark'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
