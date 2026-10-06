# Release vividRGB

The package is `vividRGB`, import `vividRGB`, repository `masahiroryo/vividRGB`.
First candidate version: `0.1.0`. A prepared distribution is not a published PyPI release.

## Validate and build

```bash
python -m pip install ".[dev,plots]"
pytest
python research/scripts/ffa_recompute_statistics.py --data research/data
python -m build
python -m twine check dist/*
```

Build a wheel and source archive. Install the wheel into a separate virtual environment,
change directory away from the checkout and run `python -c "import vividRGB"` and
`vividrgb --version`. This checks real distribution contents rather than an editable import.
CPU CI runs on Linux and Windows across Python 3.10, 3.12 and 3.13.

## Register Trusted Publishers

Sign in to each of your own PyPI and TestPyPI accounts; these are separate accounts.
Register a pending publisher for this new project using these exact settings:

| Setting | TestPyPI | PyPI |
| --- | --- | --- |
| Project name | vividRGB | vividRGB |
| GitHub owner | masahiroryo | masahiroryo |
| Repository | vividRGB | vividRGB |
| Workflow filename | publish.yml | publish.yml |
| Environment | testpypi | pypi |

Account pages: [TestPyPI publisher registration](https://test.pypi.org/manage/account/publishing/)
and [PyPI publisher registration](https://pypi.org/manage/account/publishing/).
Create repository environments named `testpypi` and `pypi`. Add a reviewer requirement
to the production `pypi` environment. No long-lived PyPI token is needed for this workflow.
The initial publication requires an actual authorized account; GitHub sign-in alone does
not create a PyPI account or grant project ownership.

## TestPyPI, then PyPI

Run the **Publish verified distributions** workflow manually with target `testpypi`.
The build job tests the source and frozen evidence; a separate job publishes those exact
artifacts through OIDC. A final job verifies their hashes and installs the package from
TestPyPI using `--no-deps`, keeping dependencies on the regular PyPI index.

After that succeeds, run the same workflow at the same commit with target `pypi`.
Production publication requires the rebuilt wheel and source archive hashes to match
TestPyPI's files. If code or metadata changed after TestPyPI publication, increment the
version and test it again; neither registry permits replacement of an uploaded file.

The workflow is deliberately manual for the first release. It can later be extended to
a GitHub release event once account configuration and version handling are established.

## GitHub release assets

Use tag `v0.1.0` at the tested commit. Upload the wheel/source archive plus all data archives
and `asset-manifest.json` listed in `research/asset-manifest.json`. Large research assets
remain outside the wheel and Git history. Each archive is under 2 GiB and contains individual
file hashes. Preserve originals and frozen descriptors, not only thumbnails or summary scores.

## Upstream instructions

This workflow follows [PyPA's packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[PyPA's GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
and [PyPI's pending-publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

