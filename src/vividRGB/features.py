"""Inference-time changes to local scale/context, with frozen encoder weights."""
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import Protocol, Sequence
import numpy as np
from PIL import Image, ImageOps
from .geometry import crop_geometry, hann_weights, resize_bilinear, sample_grid
from .projection import PCAProjection, fit_pca


class DenseEncoder(Protocol):
    """An encoder returns raw B x token_H x token_W x feature_dim descriptors."""
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray: ...


@dataclass(frozen=True)
class FocusConfig:
    focus: float = .1
    crop_size: int = 384
    overlap: float = .75
    four_flips: bool = True
    batch_size: int = 12
    hann_floor: float = .05

    def __post_init__(self):
        if not 0 < self.focus <= 1 or not 0 <= self.overlap < 1:
            raise ValueError("focus must be in (0,1]; overlap in [0,1)")
        if self.crop_size < 16 or self.crop_size % 16 or self.batch_size < 1:
            raise ValueError("crop_size must be a positive multiple of 16 and batch_size positive")
        if not 0 < self.hann_floor <= 1:
            raise ValueError("hann_floor must be in (0,1]")


@dataclass
class FeatureResult:
    features: np.ndarray
    projection: PCAProjection
    metadata: dict

    def grid(self, longest_side: int = 256):
        return sample_grid(self.features, longest_side)

    def save(self, path: str | Path):
        import json
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / 'features.npz', features=self.features)
        self.projection.save(path / 'projection.npz')
        (path / 'metadata.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')


def open_rgb(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Array inputs must be uint8 H x W x 3 RGB, values 0..255")
        return Image.fromarray(image)
    with Image.open(image) as original:
        return ImageOps.exif_transpose(original).convert('RGB')


def _encode(encoder, images):
    tokens = np.asarray(encoder.encode(images))
    if tokens.ndim != 4 or tokens.shape[0] != len(images) or not np.isfinite(tokens).all():
        raise ValueError("Encoder must return finite B x token_H x token_W x feature_dim tokens")
    return tokens


def fourflip_encode(images: Sequence[Image.Image], encoder: DenseEncoder):
    """Inverse-align four reflected raw grids, average in float32, store float16.

    The final FP16 cast matches the frozen research workflow. No descriptor L2
    normalization is introduced. PCA projection happens after the average.
    """
    accum = None
    for vertical, horizontal in [(False, False), (False, True), (True, False), (True, True)]:
        views = []
        for image in images:
            view = ImageOps.flip(image) if vertical else image
            views.append(ImageOps.mirror(view) if horizontal else view)
        z = _encode(encoder, views).astype(np.float32)
        if horizontal:
            z = z[:, :, ::-1, :]
        if vertical:
            z = z[:, ::-1, :, :]
        accum = z if accum is None else accum + z
    return (accum * .25).astype(np.float16)


def calibrate(image, encoder: DenseEncoder, n_components: int = 16,
              focuses=(.05, .10, .20, .40), tokens_per_treatment: int = 2048,
              crops_per_focus: int = 12, seed: int = 44):
    """Shared centered PCA from direct, four-flip direct, and sampled local tokens.

    Defaults preserve the six-treatment calibration in the five-domain study,
    including 20/40% calibration context even when only 5/10% outputs are shown.
    For small images, all groups use a common smaller token cap.
    """
    im = open_rgb(image)
    if tokens_per_treatment < 1 or crops_per_focus < 1:
        raise ValueError("Sampling counts must be positive")
    rng = np.random.default_rng(seed)
    direct = _encode(encoder, [im])
    reflected = fourflip_encode([im], encoder)
    cap = min(tokens_per_treatment, np.prod(direct.shape[1:3]), np.prod(reflected.shape[1:3]))
    samples = []
    for z in [direct, reflected]:
        flat = z.reshape(-1, z.shape[-1])
        samples.append(flat[rng.choice(len(flat), cap, replace=False)])
    for focus in focuses:
        _, _, boxes = crop_geometry(*im.size, focus)
        boxes = [boxes[i] for i in rng.choice(len(boxes), min(crops_per_focus, len(boxes)), replace=False)]
        views = [im.crop(box).resize((384, 384), Image.Resampling.BICUBIC) for box in boxes]
        z = fourflip_encode(views, encoder)
        flat = z.reshape(-1, z.shape[-1])
        if len(flat) < cap:
            raise ValueError("Encoder produced fewer local tokens than the calibration cap")
        samples.append(flat[rng.choice(len(flat), cap, replace=False)])
    return fit_pca(np.concatenate(samples).astype(np.float32), n_components, seed=42)


def extract_features(image, encoder: DenseEncoder, projection: PCAProjection | None = None,
                     config: FocusConfig | None = None, method: str = 'local',
                     n_components: int = 16):
    """Return dense PCA descriptors in original input-pixel coordinates.

    No vegetation mask, AnyUp, spatial-distance features, shared cluster centers,
    or scale fusion. Model loading, warm-up and PCA calibration are outside the
    extraction timer. Original working dimensions are retained.
    """
    im = open_rgb(image)
    if method not in ['local', 'dino']:
        raise ValueError("method must be local or dino")
    config = config or FocusConfig()
    projection = projection or calibrate(im, encoder, n_components)
    w, h = im.size
    sync = getattr(encoder, 'synchronize', lambda: None)
    backend = getattr(encoder, 'interpolation_backend', 'numpy')
    sync()
    begin = time.perf_counter()
    if method == 'dino':
        z = _encode(encoder, [im])
        field = resize_bilinear(projection.transform(z)[0], (h, w), backend)
        metadata = {'method': method, 'tiles': 1, 'four_flips': False}
    else:
        crop, stride, boxes = crop_geometry(w, h, config.focus, config.overlap)
        field = np.zeros((h, w, projection.basis.shape[1]), np.float32)
        weights = np.zeros((h, w, 1), np.float32)
        blend = hann_weights(crop, config.hann_floor)
        for start in range(0, len(boxes), config.batch_size):
            chunk = boxes[start:start + config.batch_size]
            crops = [im.crop(b).resize((config.crop_size, config.crop_size), Image.Resampling.BICUBIC) for b in chunk]
            z = fourflip_encode(crops, encoder) if config.four_flips else _encode(encoder, crops)
            projected = projection.transform(z)
            for pr, (x0, y0, x1, y1) in zip(projected, chunk):
                dense = resize_bilinear(pr, (crop, crop), backend)
                field[y0:y1, x0:x1] += dense * blend
                weights[y0:y1, x0:x1] += blend
        if not np.all(weights > 0):
            raise RuntimeError("Uncovered reconstruction pixels")
        field /= weights
        metadata = {'method': method, **asdict(config), 'tiles': len(boxes),
                    'crop_pixels': crop, 'stride_pixels': stride, 'actual_overlap': 1 - stride / crop}
    sync()
    metadata.update(extraction_seconds=time.perf_counter() - begin,
                    working_dimensions=[w, h], retained_dimensions=projection.basis.shape[1],
                    interpolation_backend=backend, explicit_l2_normalization=False,
                    pca_whitening=False, calibration_timed=False,
                    encoder=getattr(encoder, 'metadata', {'type': type(encoder).__name__}))
    return FeatureResult(field, projection, metadata)

