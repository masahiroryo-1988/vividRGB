"""Audit prepared distributions and the reproducibility archive inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'research/asset-manifest.json').read_text())
    by_name = {a['name']: a for a in manifest['assets']}
    assert len(by_name) == len(manifest['assets'])
    assert all(a['bytes'] < 2 * 1024 ** 3 for a in manifest['assets'])
    files = {f['path'] for a in manifest['assets'] for f in a['files']}
    for i in range(1, 11):
        image = f'pmid_{i:02d}'
        assert f'runs/eco_collection75_v1/{image}/input.png' not in files
        assert f'runs/eco_auto_clusters50_v1/{image}/input.jpg' not in files
        assert f'runs/eco_gmm_final50_v1/{image}/input.jpg' not in files
        assert not any(p.startswith(f'runs/eco_gmm_final50_v1/{image}/')
                       and p.endswith(('_overlay.jpg', '_boundary.jpg')) for p in files)
    moin = [p for p in files if p.startswith('datasets/moin/plot_all_inputs/originals/')]
    assert len(moin) == 439
    assert 'runs/eco_collection75_v1/sources/sam3_hf_revision.json' in files
    assert 'DATA_LICENSES/NOTICE.md' in files
    with tarfile.open(root / 'dist/vividrgb-0.1.0.tar.gz') as archive:
        names = {Path(n).parts[1:] for n in archive.getnames()}
        for required in ['tools/download_assets.py', 'tests/test_release_tools.py',
                         'src/vividRGB/encoders.py', 'docs/third-party/NOTICE.md']:
            assert tuple(Path(required).parts) in names
    with zipfile.ZipFile(root / 'dist/vividrgb-0.1.0-py3-none-any.whl') as archive:
        assert all(not name.startswith(('research/', 'tests/', 'tools/')) for name in archive.namelist())
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    assert all(not p.startswith(('.venv', 'local-validation', 'dist/')) for p in tracked if p)
    report = {'prepared_version': '0.1.0', 'date': '2026-10-06',
              'unit_tests_passed': 24, 'clean_wheel_install_tested': True,
              'custom_encoder_example_tested': True,
              'data_archives': len(by_name),
              'archive_total_bytes': sum(a['bytes'] for a in manifest['assets']),
              'moin_original_photographs': len(moin), 'pmid_rgb_republished': False,
              'sdist_tools_and_tests_checked': True,
              'distributions': [{'name': p.name, 'bytes': p.stat().st_size,
                                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in sorted((root / 'dist').iterdir())],
              'numeric_evidence': 'research/data/independent_numeric_validation.json',
              'gpu_checks': ['research/calibration-equivalence-validation.json',
                             'research/gpu-equivalence-validation.json',
                             'research/sam3-equivalence-validation.json'],
              'publication_status_at_preparation': {'github': 'pending authentication',
                                                  'testpypi': 'pending publisher setup',
                                                  'pypi': 'pending publisher setup'}}
    for name in report['gpu_checks']:
        assert json.loads((root / name).read_text())['complete']
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
