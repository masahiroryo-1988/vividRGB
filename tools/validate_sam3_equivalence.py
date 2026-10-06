"""Check the optional image-only adapter against one frozen SAM3 partition."""
import argparse
import json
from pathlib import Path
import numpy as np
from vividRGB.encoders import SAM3Automatic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image', default='moin_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    torch.set_num_threads(2)
    primary = args.root / 'runs/eco_collection75_v1' / args.image
    generator = SAM3Automatic(device=0)
    result = generator.generate(primary / 'input.png')
    with np.load(primary / 'sam3_hf_quality_labels.npz', allow_pickle=False) as stored:
        np.testing.assert_array_equal(result['labels'], stored['labels'])
    with np.load(primary / 'sam3_hf_proposals.npz', allow_pickle=False) as stored:
        np.testing.assert_array_equal(result['scores'], stored['quality'])
        np.testing.assert_array_equal(result['masks'].sum(axis=(1, 2)), stored['area'])
    report = {'complete': True, 'image': args.image,
              'mask_count': len(result['scores']), 'partition_exactly_equal': True,
              'scores_exactly_equal': True, 'shape': list(result['labels'].shape),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'user_supplied_prompts': result['user_supplied_prompts'],
              'scope': 'One-image image-only SAM3 adapter equivalence; not a renewed benchmark'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
