# Reproducing the research

## Levels of reproduction

1. **Saved-result verification, CPU only:** all numerical evidence is in this repository.
   Recalculate the 50-image F1 comparison, exact paired tests/Holm adjustment and cost coordinates.
2. **Frozen-feature analysis:** download the versioned clustering archive, verify its hashes,
   and rerun clustering/regularization without neural inference.
3. **Fresh neural inference:** obtain the exact working inputs and checkpoints, configure the
   recorded software and working paths, and rerun the original scripts. Timings/energy change
   with hardware and software, and GPU arithmetic is not promised to be bitwise deterministic.

The refactored library serves new photographs. The accompanying frozen research scripts
and protocols serve exact experimental provenance; results should not be silently redefined
by using later defaults or dependency versions.

## Verify the published numerical evidence

```bash
python -m pip install ".[plots]"
python research/scripts/ffa_recompute_statistics.py --data research/data
python research/scripts/ffa_cost_tradeoff.py --source research/data/ecology-collection75_results.json --out reproduction
```

This requires no model download, original RGB photographs or GPU. It validates all 200
image-method primary scores, 40 coral-reference scores, twenty paired contrasts and the
200 duration / 150 efficiency coordinates. The fifty empty DINOv3 energy fields are intentional.

## Assets and layout

Download the release assets listed in `research/asset-manifest.json` using
`python tools/download_assets.py --output research-workspace` after a GitHub release exists.
The downloader verifies each archive and each extracted file and rejects unsafe archive paths.
Archives expand beneath one root using the original relative layout:

```text
research-workspace/
  runs/eco_collection75_v1/
  runs/eco_auto_clusters50_v1/
  runs/eco_gmm_final50_v1/
  runs/moin_plot_all_20260928/
  runs/moin_phase1_benchmark_v1/
  runs/moin_phase1_fourflip_v1/
  runs/moin_multiscale_pilot_v1/
  datasets/moin/plot_all_inputs/originals/
```

MOIN development inputs are split by site to keep release assets under GitHub's per-file
limit. The frozen clustering archive contains all four historical feature methods, including
20% where required by the recorded sensitivity study. The library's main approaches remain
5% and 10%. This is preservation of experiment evidence, not a recommendation of 20%.

PMID RGB images are intentionally not republished. Retrieve them using
`python tools/fetch_pmid.py --root research-workspace`; it follows the ten frozen provider
download IDs, verifies original JPEG SHA-256 values and prepares the recorded working windows.
The required identifiers and source hashes are included in `research/data/selection_manifest.json`.
After obtaining these images, run `python tools/restore_previews.py --root research-workspace`
to reconstruct their omitted RGB overlays for the final clustering report. This uses archived
label grids and leaves fit records and numerical results unchanged. The
`final-clustering-outputs` archive preserves the final label maps and false-color previews;
`research-support` supplies per-image development metrics, script assets and license notices.

To rebuild the final inspection report without fitting models again:

```bash
export VIVIDRGB_RESEARCH_ROOT="$(pwd)/research-workspace"
python research/scripts/build_gmm_final50.py
```

## Portable research paths

`research/original-scripts/` contains byte-exact research snapshots. `research/scripts/`
contains their portable path copies; `research/script-provenance.json` records both hashes
and the complete local Python import graph. Set one environment variable:

```bash
# POSIX shell
export VIVIDRGB_RESEARCH_ROOT="$(pwd)/research-workspace"
python research/scripts/gmm_final50.py penalty --workers 2
```

```powershell
$env:VIVIDRGB_RESEARCH_ROOT = (Resolve-Path research-workspace).Path
python research/scripts/gmm_final50.py penalty --workers 2
```

Existing archived outputs are intentionally cached; move aside the specific output directory
you wish to refit before a fresh run. Do not overwrite the source evidence when comparing
different settings. K-means, GMM penalty sweeps and Bayesian concentration experiments remain
distinct. Some rendering scripts require the included HTML/JS templates and retrieved RGB inputs.

## Environments

`research/requirements-cpu.txt` pins the recorded scientific analysis environment.
`research/requirements-inference.txt` adds the recorded PyTorch/Transformers stack.
These describe the Linux GPU research environment, not a guaranteed Windows CUDA installer.
Install a PyTorch build suitable for your platform according to its official instructions.
Protocol JSON records exact checkpoint revisions and sampling seeds. Current dependency
ranges in `pyproject.toml` support the installable library and are deliberately separate
from archived research-version pins.

## Research scripts and checks

The dependency closure includes phase-one/four-flip development, scale pilot, five-domain
DINO and automatic SAM3 comparison, frozen feature extraction, automatic clustering,
penalty/prior sweeps, varimax/stability checks, result aggregation and figures. SHA-256
provenance is recorded for inputs, projections, script snapshots and frozen results.

Raw per-image timings are observational records and cannot be reproduced by recomputing
the same number from a different GPU. Regenerate them and label the hardware/stages rather
than treating agreement with the historical cost as a correctness test.

## Library equivalence checks

The release includes three one-image GPU validation records. With the archived working
MOIN input and pinned environments, fresh library PCA calibration, the 10% reconstructed
256 × 256 × 16 grid, and the image-only SAM3 partition and scores exactly matched the saved
outputs. This checks implementation equivalence for that example; it does not establish
general ecological accuracy or bitwise agreement on other hardware. The corresponding
validation scripts are in `tools/`; SAM3 uses its separate recorded Transformers 5 environment.
