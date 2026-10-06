# vividRGB

Discover dense visual cues in ecological photographs with **forced focused attention**.
The workflow re-presents local crops to a frozen vision encoder, averages four inverse-aligned
reflections, and reconstructs an overlapping feature field. PCA and orthogonal varimax make
appearance contrasts visible; independent K-means and regularized full-covariance Gaussian
mixtures summarize them as visual groups.

**First release: 0.1.0, alpha.** Research implementation with an installable Python API and CLI.
Model-agnostic interfaces are supported; the research evaluation tested DINOv3. This software
does not change the transformer's attention architecture or weights.

[Demonstrator](https://kics-zert.projectcove.org/) ·
[Five-domain report](https://kics-zert.projectcove.org/reports/5_dataset_analysis/) ·
[Methods and evaluation](https://kics-zert.projectcove.org/technology/#f1-benchmark)

## Install

From a downloaded wheel or a local clone:

```bash
python -m pip install .
# Frozen DINOv3 inference (PyTorch/Transformers are optional):
python -m pip install ".[dino]"
```

After PyPI publication, the equivalent command is `pip install "vividRGB[dino]"`.
Upstream DINOv3 and SAM3 model downloads require accepting their own model access terms.
Use `hf auth login` or the upstream Hugging Face authentication mechanism; never commit credentials.

## Python example

```python
from PIL import Image
from vividRGB import FocusConfig, calibrate, extract_features, varimax, pca_rgb, cluster_gmm
from vividRGB.encoders import DINOv3Encoder

encoder = DINOv3Encoder(device="cuda")
encoder.warm_up()
# Prepare a working image whose sides are multiples of 16; no implicit resizing is performed.
image = Image.open("working-image.png").convert("RGB")
projection = calibrate(image, encoder)
result = extract_features(image, encoder, projection, FocusConfig(focus=0.10))
Image.fromarray(pca_rgb(result.features, limits=(projection.low, projection.high))).save("pca.png")
rotation = varimax(projection.basis)["rotation"]
Image.fromarray(pca_rgb(result.features, rotation=rotation)).save("varimax.png")
grid, _, _ = result.grid(256)
clusters = cluster_gmm(grid, penalty=4)
print(clusters.metadata["selected_k"])
```

Use the **same frozen per-image projection** for comparing DINOv3, 5% focus and 10% focus.
Cluster models are fitted independently; they do not use false-color RGB or spatial coordinates.
Core CPU operations and custom encoders work without installing PyTorch.

```bash
vividrgb analyze working-image.png --focus 0.10 --algorithm gmm --penalty 4 --output analysis
vividrgb analyze working-image.png --focus 0.05 --projection analysis/projection.npz --output focus5
vividrgb analyze working-image.png --method dino --projection analysis/projection.npz --output dino
```

## Reproduce the research

The repository contains `research/data/` frozen results, protocols, selections and SHA-256
values, plus the complete dependency closure of the research scripts. Large images and frozen
features are kept as versioned release assets rather than inside the Python wheel.
See [reproduction instructions](docs/reproducibility.md), [data licensing](docs/data-licenses.md)
and [scientific scope](docs/methods.md).

```bash
python -m pip install ".[plots]"
python research/scripts/ffa_recompute_statistics.py --data research/data
python research/scripts/ffa_cost_tradeoff.py --source research/data/ecology-collection75_results.json --out reproduction
```

The first command verifies all 200 primary RGB-edge F1 values, 40 coral annotated scores,
paired contrasts and Figure 7 coordinates from saved evidence, without a GPU or source photographs.
The second regenerates the performance-versus-duration/energy figure.

**Interpretation:** primary F1 is strong-RGB-edge alignment, an appearance proxy. It is not
leaf/species segmentation accuracy or biological diversity. SAM3 has the highest annotated
coral class-boundary F1 in the saved comparison. Energy is estimated GPU board energy, with
differing implementation stage definitions; full-stage DINOv3 energy is unavailable.

## Develop and release

```bash
python -m pip install ".[dev,plots]"
pytest
python -m build
python -m twine check dist/*
```

[Release instructions](docs/releasing.md) describe TestPyPI, clean wheel installation and
PyPI Trusted Publishing through GitHub Actions. No model weights, photographs, credentials
or server-specific paths are bundled in the library wheel. MIT license for original code.

