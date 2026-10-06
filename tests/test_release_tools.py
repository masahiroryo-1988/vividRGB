"""Exercise archive validation used for the public research download."""
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location(
    'download_assets', Path(__file__).resolve().parents[1] / 'tools/download_assets.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def archive_with(name, kind=tarfile.REGTYPE):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            member.linkname = '../outside.txt'
        else:
            member.size = 4
        archive.addfile(member, io.BytesIO(b'data') if kind == tarfile.REGTYPE else None)
    data.seek(0)
    return tarfile.open(fileobj=data)


@pytest.mark.parametrize('name', ['../outside.txt', '/absolute.txt', r'..\outside.txt',
                                 'C:outside.txt', 'image.png:stream'])
def test_archive_rejects_paths_before_extracting(tmp_path, name):
    with archive_with(name) as archive:
        with pytest.raises(ValueError, match='Unsafe archive member'):
            download.safe_extract(archive, tmp_path / 'destination')
    assert not list((tmp_path / 'destination').iterdir())


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_archive_rejects_links(tmp_path, kind):
    with archive_with('link', kind) as archive:
        with pytest.raises(ValueError, match='Unsafe archive member'):
            download.safe_extract(archive, tmp_path)
    assert not list(tmp_path.iterdir())


def test_archive_extracts_nested_regular_file(tmp_path):
    with archive_with('runs/image/feature.npz') as archive:
        download.safe_extract(archive, tmp_path)
    assert (tmp_path / 'runs/image/feature.npz').read_bytes() == b'data'
