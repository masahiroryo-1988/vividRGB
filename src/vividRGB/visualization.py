"""False color displays are visual descriptors, not RGB reconstruction or taxa."""
import numpy as np


def pca_rgb(features, rotation=None, limits=None, percentiles=None):
    x = np.asarray(features)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite H x W x dimensions descriptors')
    if rotation is not None:
        r = np.asarray(rotation)
        if r.shape != (x.shape[-1], x.shape[-1]):
            raise ValueError('Rotation must cover all retained dimensions')
        x = x @ r
    shown = x[..., :3]
    if limits is None:
        low, high = np.percentile(shown.reshape(-1, shown.shape[-1]),
                                  percentiles or ((2, 98) if rotation is not None else (1, 99)), axis=0)
    else:
        low, high = limits
    scaled = np.clip((shown - low) / np.maximum(np.asarray(high) - low, 1e-6), 0, 1)
    if shown.shape[-1] < 3:
        scaled = np.pad(scaled, ((0, 0), (0, 0), (0, 3 - shown.shape[-1])))
    return np.uint8(scaled * 255)


def labels_rgb(labels, seed=33):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Labels must be an integer H x W array')
    values, inverse = np.unique(labels, return_inverse=True)
    palette = np.random.default_rng(seed).integers(25, 240, (len(values), 3), dtype=np.uint8)
    palette[values < 0] = 0
    return palette[inverse].reshape(*labels.shape, 3)

