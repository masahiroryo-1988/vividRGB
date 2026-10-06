"""Read the version from the actual wheel, including in release workflows."""
import argparse
from email.parser import BytesParser
import os
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    parser.add_argument('--github-output', action='store_true')
    args = parser.parse_args()
    wheels = list(args.dist.glob('*.whl'))
    sources = list(args.dist.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError('Expected exactly one wheel and one source archive')
    with zipfile.ZipFile(wheels[0]) as archive:
        candidates = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(candidates) != 1:
            raise RuntimeError('Ambiguous wheel metadata')
        metadata = BytesParser().parsebytes(archive.read(candidates[0]))
    if metadata['Name'].lower() != 'vividrgb':
        raise RuntimeError('Unexpected package in release artifacts')
    version = metadata['Version']
    if not version or '\n' in version or '\r' in version:
        raise RuntimeError('Invalid distribution version')
    if args.github_output:
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write(f'version={version}\n')
    print(version)


if __name__ == '__main__':
    main()
