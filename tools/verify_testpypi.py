"""Require the exact built artifacts to be present on TestPyPI before production publication."""
import argparse
import hashlib
import json
import time
import urllib.request
import urllib.error
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='vividRGB')
    parser.add_argument('--version', required=True)
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    args = parser.parse_args()
    url = f'https://test.pypi.org/pypi/{args.name}/{args.version}/json'
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.load(response)
            break
        except urllib.error.HTTPError:
            if attempt == 3:
                raise
            time.sleep(10)
    upstream = {file['filename']: file['digests']['sha256'] for file in metadata['urls']}
    distributions = [p for p in args.dist.iterdir() if p.suffix == '.whl' or p.name.endswith('.tar.gz')]
    if len(distributions) != 2:
        raise RuntimeError('Expected one wheel and one source archive')
    for path in distributions:
        if upstream.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f'TestPyPI does not contain the exact candidate artifact: {path.name}')
    print('Verified exact wheel and source archive on TestPyPI')


if __name__ == '__main__':
    main()

