import numpy as np
import pytest
from PIL import Image
from vividRGB import FocusConfig, PCAProjection, extract_features, fourflip_encode, fit_pca
from vividRGB.geometry import starts, crop_geometry, resize_bilinear, sample_grid


class RGBEncoder:
    """Equivariant test double for alignment tests; not a scientific benchmark."""
    def encode(self, images):
        return np.stack([np.asarray(i).astype(np.float32) / 255 for i in images])


@pytest.mark.parametrize('size', [(64, 64), (97, 129), (32, 400)])
def test_edge_aligned_windows_cover_every_pixel(size):
    crop, stride, boxes = crop_geometry(*size, focus=.10)
    coverage = np.zeros(size[::-1], bool)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] = True
    assert coverage.all()
    assert boxes[-1][2:] == size
    assert stride == max(1, round(crop / 4))


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        starts(10, 20, 5)
    with pytest.raises(ValueError):
        FocusConfig(overlap=1)


def test_inverse_aligned_reflections_preserve_an_equivariant_field():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (16, 32, 3), dtype=np.uint8))
    original = RGBEncoder().encode([image]).astype(np.float16)
    np.testing.assert_array_equal(fourflip_encode([image], RGBEncoder()), original)


def test_positive_hann_coverage_preserves_constant_descriptors():
    im = Image.new('RGB', (37, 51), (64, 128, 192))
    projection = PCAProjection(np.eye(3), np.zeros(3))
    result = extract_features(im, RGBEncoder(), projection, FocusConfig(crop_size=16))
    expected = (np.array([64, 128, 192]) / 255).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(result.features, np.broadcast_to(expected, result.features.shape), atol=1e-6)
    assert result.metadata['actual_overlap'] == .75
    assert result.metadata['four_flips']


def test_bilinear_half_pixel_coordinates_and_edges():
    grid = np.array([[[0.], [4.]], [[8.], [12.]]], dtype=np.float32)
    out = resize_bilinear(grid, (4, 4))[..., 0]
    np.testing.assert_allclose(out, [[0, 1, 3, 4], [2, 3, 5, 6], [6, 7, 9, 10], [8, 9, 11, 12]])


def test_grid_samples_cell_centers_without_averaging():
    field = np.arange(4 * 8 * 2).reshape(4, 8, 2)
    grid, ys, xs = sample_grid(field, 4)
    np.testing.assert_array_equal(ys, [1, 3])
    np.testing.assert_array_equal(xs, [1, 3, 5, 7])
    np.testing.assert_array_equal(grid, field[ys[:, None], xs[None, :]])


def test_pca_roundtrip_preserves_raw_projection(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(100, 6)).astype(np.float32)
    x[:, 0] *= 8
    projection = fit_pca(x, n_components=3)
    path = tmp_path / 'pca.npz'
    projection.save(path)
    loaded = PCAProjection.load(path)
    np.testing.assert_array_equal(projection.transform(x), loaded.transform(x))
    # First PC variance remains larger; no implicit whitening.
    assert np.var(projection.transform(x)[:, 0]) > 10 * np.var(projection.transform(x)[:, 2])

