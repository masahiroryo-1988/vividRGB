# Using vividRGB

The package import is case-sensitive: `import vividRGB`. The command-line executable
is lowercase: `vividrgb`. PyPI normalizes the distribution name to `vividrgb`.

## Input images and encoder access

Use RGB working images whose sides are multiples of 16 for the pinned DINOv3 backend.
The library deliberately performs no hidden thumbnailing, padding, vegetation masking,
or global enlargement. `open_rgb(path)` applies EXIF orientation; an already-open PIL
image is treated as the intended working image. Record any working crop you prepare.
Crop percentages refer to **side length**, not area, relative to the short image side.

Install `vividRGB[dino]` for the frozen DINOv3 adapter. On CUDA, pass a batch size suitable
for your GPU; twelve 384-pixel crops per orientation matches the original run. A 2048-pixel
image at 5% focus may take minutes and use substantial memory. On CPU the encoder forward
pass uses FP32; the reported benchmark used CUDA FP16 and does not establish CPU equivalence.
The returned raw token grids are stored in FP16 to match the research storage convention.
Accept the checkpoint's own access/license terms and authenticate with Hugging Face before
downloading. `local_files_only=True` supports an already populated checkpoint cache.

## Compare the same image fairly

1. Prepare a working image once.
2. Fit `calibrate(image, encoder)` once, or load its saved `PCAProjection`.
3. Extract DINOv3, local 5%, and local 10% fields with that same projection.
4. Fit each method's cluster model independently on raw PCA coordinates.
5. Use compatible display limits; each method may otherwise look different merely because
   its color contrast was stretched independently.

The default calibration uses direct and four-flip direct tokens plus sampled 5%, 10%, 20%
and 40% crops, retaining the six-treatment basis of the original study. Twenty/forty percent
contribute calibration samples; they are not fused into the selected 5/10% feature field.
Small inputs use fewer common tokens per calibration treatment. `n_components=16` requires
at least sixteen samples and sixteen encoder channels.

## Clustering

`cluster_kmeans(grid, k="auto")` uses K=3..12, held-out silhouette on three spatial-block
folds, and chooses the candidate closest to five within one standard error of the best.
`cluster_kmeans(field, k=6)` reproduces the primary benchmark's fixed-K design; use the
full-resolution field, not a 256-side grid, for the original F1 sampling domain.

`cluster_gmm(grid, penalty=4)` fits K=1..12 full-covariance MLE mixtures and minimizes
`-2 log L + lambda p_K log n`. It uses covariance regularization `1e-5`, three starts,
seed 42 and a final sample of at most 8192 locations. Lambda=1 is ordinary BIC. Larger
lambda can reduce the selected count but does not guarantee a target count. A result at
K=12 means the search limit is active. Model responsibilities are not ecological confidence.

Both methods use original retained descriptors. They do not cluster the first three
display colors, add XY distances, or enforce equal pixel areas. Full-dimensional orthogonal
varimax preserves distances and full-covariance model information, although independent
refits can differ through initialization and numerical precision.

## Custom encoders

Provide an object with `encode(images)` returning a NumPy array with shape
`batch x token_height x token_width x feature_dimension`. All input images in one batch
have the same size. Feature maps must use the image's horizontal and vertical axes.
Optionally expose `metadata`, `synchronize()` and `interpolation_backend` (`numpy` or `torch`).
The NumPy reconstruction backend requires no Torch installation. Half-pixel bilinear
interpolation and positive Hann coverage are used in either backend.

The `examples/custom_encoder.py` RGB descriptor is an API illustration and unit-test aid;
it is not an ecological evaluation model. The model-agnostic interface does not establish
that the method improves every backbone: the empirical study used DINOv3.

## Outputs

The CLI saves dense descriptors, the sampled grid and coordinates, the reusable PCA
projection, varimax rotation, PCA and PCA-varimax PNGs, labels and metadata. It saves
categorical maps with nearest-neighbor enlargement; descriptor displays use continuous
color channels. Duration excludes model loading, warm-up and PCA calibration.

## SAM3

`vividRGB.encoders.SAM3Automatic` uses the Hugging Face image-only mask-generation pipeline,
the pinned SAM3 revision, FP32 and no human-supplied prompts. The pipeline internally
generates a point grid; image-only input does not mean no internal prompts. The highest
predicted mask IoU wins overlaps, with background label zero. This adapter returns masks
and a partition, not a DINO-compatible embedding for forced focused attention.

