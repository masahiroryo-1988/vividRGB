"""Download and verify versioned research archives; never extract unsafe paths."""
import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    members = archive.getmembers()
    for member in members:
        p = PurePosixPath(member.name)
        target = root.joinpath(*p.parts).resolve()
        if (p.is_absolute() or '..' in p.parts or '\\' in member.name or ':' in member.name
                or not target.is_relative_to(root) or not (member.isfile() or member.isdir())):
            raise ValueError(f'Unsafe archive member: {member.name}')
    archive.extractall(root, members=members, filter='data') if hasattr(tarfile, 'data_filter') else archive.extractall(root, members=members)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'research/asset-manifest.json')
    parser.add_argument('--output', type=Path, default=Path('research-workspace'))
    parser.add_argument('--assets', nargs='*', help='Selected asset filenames; default all')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--base-url', default='https://github.com/masahiroryo-1988/vividRGB/releases/download/v0.1.0/')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.list:
        for asset in manifest['assets']:
            print(asset['name'], asset['bytes'], 'bytes;', asset['file_count'], 'files')
        return
    available = {a['name'] for a in manifest['assets']}
    if args.assets and not set(args.assets) <= available:
        parser.error('Unknown asset filename')
    cache = args.output / '.archives'
    cache.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        if args.assets and asset['name'] not in args.assets:
            continue
        path = cache / asset['name']
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(args.base_url.rstrip('/') + '/' + asset['name'], timeout=60) as response, partial.open('wb') as f:
                for chunk in iter(lambda: response.read(2 ** 20), b''):
                    f.write(chunk)
            partial.replace(path)
        if path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
            raise RuntimeError(f'Archive digest mismatch: {path}')
        with tarfile.open(path) as archive:
            safe_extract(archive, args.output)
        for file in asset['files']:
            extracted = args.output / file['path']
            if extracted.stat().st_size != file['bytes'] or digest(extracted) != file['sha256']:
                raise RuntimeError(f'Extracted file digest mismatch: {extracted}')
        print('Verified', asset['name'], flush=True)


if __name__ == '__main__':
    main()
