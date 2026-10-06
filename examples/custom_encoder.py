"""Executable API example with a simple RGB descriptor, not a scientific comparator."""
import numpy as np
from PIL import Image
from vividRGB import FocusConfig, PCAProjection, extract_features, pca_rgb


class RGBDescriptor:
    def encode(self, images):
        return np.stack([np.asarray(im, dtype=np.float32) / 255 for im in images])


if __name__ == '__main__':
    rng = np.random.default_rng(2)
    image = Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    result = extract_features(image, RGBDescriptor(), PCAProjection(np.eye(3), np.zeros(3)),
                              FocusConfig(crop_size=16))
    print('Feature shape:', result.features.shape)
    print('Color-map shape:', pca_rgb(result.features).shape)

