"""Retrieve the ten frozen PMID photographs directly from their original provider."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'research/data/selection_manifest.json')
    parser.add_argument('--originals', type=Path, help='Use an existing directory of provider JPEGs instead of downloading')
    args = parser.parse_args()
    rows = [r for r in json.loads(args.manifest.read_text(encoding='utf-8')) if r['dataset'] == 'pmid']
    assert len(rows) == 10
    for row in rows:
        dest = args.root / 'runs/eco_collection75_v1' / row['id']
        dest.mkdir(parents=True, exist_ok=True)
        original = (args.originals / row['source_member'] if args.originals else
                    args.root / 'runs/eco_collection75_v1/originals/pmid' / row['source_member'])
        original.parent.mkdir(parents=True, exist_ok=True)
        if not original.exists():
            try:
                import gdown
            except ImportError as error:
                raise RuntimeError('Install gdown (pip install gdown) or pass --originals with provider JPEGs') from error
            gdown.download(id=row['download_id'], output=str(original), quiet=False)
        if not original.is_file() or sha(original) != row['source_sha256']:
            raise RuntimeError(f'Original source SHA-256 mismatch: {row["id"]}')
        with Image.open(original) as im:
            working = im.convert('RGB').crop(row['native_window'])
            working.save(dest / 'input.png')
            # The original JPEG digest is invariant; PNG compression may differ by Pillow build.
            if sha(dest / 'input.png') != row['input_sha256']:
                print('NOTE:', row['id'], 'PNG byte hash differs. Use recorded Pillow version for byte-exact input files.')
            thumb = working.copy()
            thumb.thumbnail((640, 640))
            thumb.save(dest / 'original.jpg', quality=93)
            frozen = args.root / 'runs/eco_auto_clusters50_v1' / row['id']
            frozen.mkdir(parents=True, exist_ok=True)
            preview = working.copy()
            preview.thumbnail((1024, 1024))
            preview.save(frozen / 'input.jpg', quality=94)
        print('Verified original source and prepared', row['id'])


if __name__ == '__main__':
    main()

