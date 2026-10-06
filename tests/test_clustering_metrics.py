import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from vividRGB import varimax, cluster_gmm, cluster_kmeans, pca_rgb
from vividRGB.clustering import gmm_parameter_count
from vividRGB.metrics import boundary_f1, boundaries, rgb_edge_f1


def features():
    rng = np.random.default_rng(13)
    x = rng.normal(scale=.2, size=(24, 24, 3))
    x[:12, :12] += [-4, 0, 0]
    x[:12, 12:] += [4, 0, 0]
    x[12:] += [0, 4, 0]
    return x.astype(np.float32)


def test_varimax_preserves_distances_and_projection_subspace():
    rng = np.random.default_rng(22)
    basis = np.linalg.qr(rng.normal(size=(40, 6)))[0]
    result = varimax(basis)
    rotation = result['rotation']
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(basis @ rotation @ (basis @ rotation).T, basis @ basis.T, atol=1e-12)
    x = rng.normal(size=(30, 6))
    np.testing.assert_allclose(np.linalg.norm(x - x[::-1], axis=1),
                               np.linalg.norm(x @ rotation - x[::-1] @ rotation, axis=1), atol=1e-12)
    assert result['objective'] >= result['initial_objective'] - 1e-8


def test_fixed_kmeans_has_full_coverage_without_equal_area_constraint():
    result = cluster_kmeans(features(), k=3)
    assert result.labels.shape == (24, 24)
    assert result.metadata['occupied_k'] == 3
    truth = np.full((24, 24), 2)
    truth[:12, :12], truth[:12, 12:] = 0, 1
    assert adjusted_rand_score(result.labels.ravel(), truth.ravel()) > .98
    areas = np.sort(np.bincount(result.labels.ravel()))
    assert areas[-1] > 1.5 * areas[0]


def test_auto_kmeans_reports_candidate_scores_and_selection_rule():
    result = cluster_kmeans(features(), candidate_k=range(3, 6))
    assert result.metadata['selected_k'] in result.metadata['eligible_k']
    assert result.metadata['selected_k'] == min(result.metadata['eligible_k'], key=lambda k: (abs(k - 5), k))
    assert len(result.metadata['candidates']) == 3


def test_lambda_one_equals_model_bic_and_lambda_four_uses_full_parameter_count():
    field = features()
    result = cluster_gmm(field, penalty=1, candidate_k=range(1, 5), n_init=1)
    candidate = next(c for c in result.metadata['candidates'] if c['k'] == result.metadata['selected_k'])
    # All 576 cells are retained in the final sample, so BIC is order-independent.
    np.testing.assert_allclose(candidate['criterion'], result.model.bic(field.reshape(-1, 3).astype(np.float64)), rtol=1e-10)
    assert gmm_parameter_count(3, 3) == 29
    strong = cluster_gmm(field, penalty=64, candidate_k=range(1, 5), n_init=1)
    assert strong.metadata['selected_k'] <= result.metadata['selected_k']
    assert strong.labels.min() >= 0


def test_zero_tolerance_is_exact_not_dilation_until_stable():
    a, b = np.zeros((8, 8), bool), np.zeros((8, 8), bool)
    a[3, 3], b[3, 4] = True, True
    assert boundary_f1(a, b, tolerance=0)['f1'] == 0
    assert boundary_f1(a, b, tolerance=1)['f1'] == 1


def test_boundary_orientation_and_proxy_empty_case():
    labels = np.zeros((16, 16), int)
    labels[:, 8:] = 1
    assert boundaries(labels)[:, 7].all()
    assert boundaries(labels).sum() == 16
    assert rgb_edge_f1(np.zeros_like(labels), np.zeros((16, 16, 3), np.uint8))['f1'] == 0


def test_three_channel_display_is_not_the_clustering_input():
    x = features()
    colors = pca_rgb(x)
    assert colors.shape == (24, 24, 3) and colors.dtype == np.uint8
    with pytest.raises(ValueError):
        cluster_kmeans(np.zeros((3, 3)), k=3)
