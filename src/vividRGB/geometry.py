"""Window geometry and source-coordinate interpolation; no XY feature augmentation."""
import numpy as np


def starts(length: int, crop: int, stride: int) -> list[int]:
    if length < 1 or crop < 1 or crop > length or stride < 1 or stride > crop:
        raise ValueError("Require 1 <= stride <= crop <= length")
    return sorted(set(list(range(0, length - crop + 1, stride)) + [length - crop]))


def crop_geometry(width: int, height: int, focus: float = .1, overlap: float = .75):
    if min(width, height) < 16:
        raise ValueError("The input short side must be at least 16 pixels")
    if not 0 < focus <= 1 or not 0 <= overlap < 1:
        raise ValueError("focus must be in (0, 1]; overlap in [0, 1)")
    crop = max(16, round(min(width, height) * focus))
    stride = max(1, round(crop * (1 - overlap)))
    boxes = [(x, y, x + crop, y + crop)
             for y in starts(height, crop, stride) for x in starts(width, crop, stride)]
    return crop, stride, boxes


def hann_weights(crop: int, floor: float = .05) -> np.ndarray:
    if crop < 1 or not 0 < floor <= 1:
        raise ValueError("Require positive crop and floor in (0, 1]")
    axis = np.maximum(np.hanning(crop), floor)
    return (axis[:, None] * axis[None, :])[..., None].astype(np.float32)


def resize_bilinear(grid: np.ndarray, shape: tuple[int, int], backend: str = "numpy"):
    """Half-pixel interpolation with edge clamping, equivalent to align_corners=False."""
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 3 or min(*shape, *grid.shape[:2]) < 1:
        raise ValueError("Expected a nonempty H x W x C grid and positive target shape")
    if backend == "torch":
        import torch
        import torch.nn.functional as F
        z = torch.from_numpy(np.ascontiguousarray(grid)).permute(2, 0, 1)[None]
        return F.interpolate(z, size=shape, mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    if backend != "numpy":
        raise ValueError("Interpolation backend must be numpy or torch")
    h, w = shape
    sy = np.clip((np.arange(h, dtype=np.float32) + .5) * (grid.shape[0] / h) - .5, 0, grid.shape[0] - 1)
    sx = np.clip((np.arange(w, dtype=np.float32) + .5) * (grid.shape[1] / w) - .5, 0, grid.shape[1] - 1)
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    y1, x1 = np.minimum(y0 + 1, grid.shape[0] - 1), np.minimum(x0 + 1, grid.shape[1] - 1)
    wy, wx = (sy - y0).astype(np.float32)[:, None, None], (sx - x0).astype(np.float32)[None, :, None]
    upper = grid[y0[:, None], x0[None, :]] * (1 - wx) + grid[y0[:, None], x1[None, :]] * wx
    lower = grid[y1[:, None], x0[None, :]] * (1 - wx) + grid[y1[:, None], x1[None, :]] * wx
    return upper * (1 - wy) + lower * wy


def sample_grid(field: np.ndarray, longest_side: int = 256):
    """Sample pixel centers on an aspect-preserving grid; not cell averages."""
    if field.ndim != 3 or longest_side < 1:
        raise ValueError("Expected H x W x C features and a positive grid side")
    h, w = field.shape[:2]
    gh, gw = max(1, round(h * longest_side / max(h, w))), max(1, round(w * longest_side / max(h, w)))
    ys = np.minimum(((np.arange(gh) + .5) * h / gh).astype(int), h - 1)
    xs = np.minimum(((np.arange(gw) + .5) * w / gw).astype(int), w - 1)
    return field[ys[:, None], xs[None, :]].copy(), ys, xs

