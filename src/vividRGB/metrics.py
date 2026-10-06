"""Boundary scores: appearance proxies and reference alignment, never inferred taxa."""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, binary_erosion


def boundaries(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('Labels must be H x W')
    result = np.zeros(labels.shape, bool)
    result[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    result[:-1] |= labels[:-1] != labels[1:]
    return result


def boundary_f1(pred, reference, valid=None, tolerance=2):
    pred, reference = np.asarray(pred, bool), np.asarray(reference, bool)
    valid = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    if pred.shape != reference.shape or valid.shape != pred.shape or pred.ndim != 2:
        raise ValueError('Boundary and validity masks must have identical H x W shape')
    if not isinstance(tolerance, (int, np.integer)) or tolerance < 0:
        raise ValueError('Tolerance must be a nonnegative integer')
    pred, reference = pred & valid, reference & valid
    # scipy iterations=0 means dilation until stable, so zero tolerance needs an explicit branch.
    near_ref = binary_dilation(reference, iterations=tolerance) if tolerance else reference
    near_pred = binary_dilation(pred, iterations=tolerance) if tolerance else pred
    precision = float((pred & near_ref).sum() / pred.sum()) if pred.any() else 0.
    recall = float((reference & near_pred).sum() / reference.sum()) if reference.any() else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'precision': precision, 'recall': recall, 'f1': f1}


def rgb_edges(rgb, valid, quantile=.85):
    rgb, valid = np.asarray(rgb), np.asarray(valid, bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or valid.shape != rgb.shape[:2] or not valid.any():
        raise ValueError('Expected RGB image and matching nonempty validity mask')
    if not 0 < quantile < 1:
        raise ValueError('Quantile must be in (0,1)')
    gray = gaussian_filter(rgb.astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32) / 255, 1)
    gradient = np.hypot(sobel(gray, axis=1) / 8, sobel(gray, axis=0) / 8)
    threshold = max(1e-6, float(np.quantile(gradient[valid], quantile)))
    return gradient >= threshold, gradient


def rgb_edge_f1(labels, rgb, tolerance=2):
    valid = binary_erosion(np.ones(np.asarray(labels).shape, bool), iterations=2, border_value=0)
    edges, _ = rgb_edges(rgb, valid)
    return boundary_f1(boundaries(labels), edges, valid, tolerance)

