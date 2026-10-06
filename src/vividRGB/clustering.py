"""Independent partitions; no equal-area constraints or spatial coordinates in distances."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: object
    metadata: dict
    confidence: np.ndarray | None = None


def _matrix(features, dtype=np.float64):
    features = np.asarray(features)
    if features.ndim != 3 or not np.isfinite(features).all() or min(features.shape) < 1:
        raise ValueError('Expected finite H x W x retained_dim features')
    return np.ascontiguousarray(features.reshape(-1, features.shape[-1]), dtype=dtype), features.shape[:2]


def spatial_partitions(shape):
    """Three 8 x 8 spatial-block folds and the original fixed final sample."""
    h, w = shape
    y, x = np.indices(shape)
    blocks = (np.minimum(y * 8 // h, 7) * 8 + np.minimum(x * 8 // w, 7)).ravel()
    rng = np.random.default_rng(240103)
    order = rng.permutation(64)
    fold = np.empty(64, int)
    fold[order] = np.arange(64) % 3
    splits = []
    for index in range(3):
        train, test = np.flatnonzero(fold[blocks] != index), np.flatnonzero(fold[blocks] == index)
        splits.append((rng.choice(train, min(3072, len(train)), False),
                       rng.choice(test, min(1024, len(test)), False)))
    final = rng.choice(h * w, min(8192, h * w), False)
    return splits, final


def _predict(model, x, shape, batch=65536):
    labels = np.empty(len(x), np.int32)
    for start in range(0, len(x), batch):
        labels[start:start + batch] = model.predict(x[start:start + batch])
    return labels.reshape(shape)


def cluster_kmeans(features, k: int | str = 'auto', candidate_k=range(3, 13),
                   preferred_k: int = 5, seed: int = 42, max_fit_samples: int = 8192):
    """Fixed K, or blocked silhouette + one-SE selection favoring K near five.

    Auto selection uses K=3..12 by default, common samples for every candidate,
    three folds, and a final 8192-location fit. These are explicit preferences,
    not species-count inference. Fixed K=6 is the primary boundary benchmark.
    """
    x, shape = _matrix(features, dtype=np.float32)
    if max_fit_samples < 1:
        raise ValueError('max_fit_samples must be positive')
    metadata = {'algorithm': 'kmeans', 'independent_fit': True, 'spatial_coordinates_in_features': False}
    with threadpool_limits(limits=2):
        if k == 'auto':
            splits, final = spatial_partitions(shape)
            candidates = sorted(set(candidate_k))
            if not candidates or any(not isinstance(i, (int, np.integer)) or i < 2 for i in candidates):
                raise ValueError('Automatic K-means candidates must be integers >= 2')
            if any(min(len(a), len(b)) <= max(candidates) for a, b in splits):
                raise ValueError('The grid is too small for these K candidates and spatial folds')
            scores = []
            for value in candidates:
                folds = []
                for f, (train, test) in enumerate(splits):
                    model = KMeans(n_clusters=value, n_init=3, random_state=seed + f).fit(x[train])
                    lab = model.predict(x[test])
                    folds.append(float(silhouette_score(x[test], lab))
                                 if 1 < len(np.unique(lab)) < len(test) else np.nan)
                scores.append(folds)
            scores = np.asarray(scores)
            valid = np.isfinite(scores).all(1)
            if not valid.any():
                raise RuntimeError('No candidate has valid held-out silhouette scores')
            means = np.where(valid, scores.mean(1), -np.inf)
            best = int(means.argmax())
            se = float(scores[best].std(ddof=1) / np.sqrt(3))
            eligible = [value for i, value in enumerate(candidates) if valid[i] and means[i] >= means[best] - se - 1e-12]
            selected = min(eligible, key=lambda value: (abs(value - preferred_k), value))
            metadata.update(candidates=[{'k': value, 'fold_scores': [float(v) if np.isfinite(v) else None for v in scores[i]]}
                                        for i, value in enumerate(candidates)],
                            best_score_k=candidates[best], eligible_k=eligible, one_se_tolerance=se,
                            preferred_k=preferred_k, folds=3, uncertainty_note='Spatial folds remain dependent through overlapping encoder context')
        else:
            if not isinstance(k, (int, np.integer)) or not 1 <= k <= min(max_fit_samples, len(x)):
                raise ValueError('k must be auto or a positive integer no larger than the fit sample')
            selected = int(k)
            final = np.random.default_rng(45).choice(len(x), min(max_fit_samples, len(x)), False)
        model = KMeans(n_clusters=selected, n_init=10, random_state=seed).fit(x[final])
        labels = _predict(model, x, shape)
    metadata.update(selected_k=selected, occupied_k=int(len(np.unique(labels))),
                    fit_sample_count=len(final), inertia=float(model.inertia_))
    return ClusterResult(labels, model, metadata)


def gmm_parameter_count(k: int, dimensions: int) -> int:
    return k * (dimensions + dimensions * (dimensions + 1) // 2) + k - 1


def cluster_gmm(features, penalty: float = 4., candidate_k=range(1, 13),
                reg_covar: float = 1e-5, seed: int = 42, n_init: int = 3,
                max_iter: int = 300, tol: float = .001):
    """Select a full-covariance MLE GMM by -2 log L + lambda p_K log n.

    Lambda=1 is ordinary BIC; lambda=4 is the retained exploratory setting.
    Lambda is a complexity multiplier, not a Bayesian concentration prior.
    Scores must not be compared between different retained dimensions.
    """
    x, shape = _matrix(features)
    if penalty <= 0 or reg_covar <= 0 or n_init < 1 or max_iter < 1 or tol <= 0:
        raise ValueError('GMM regularization, iterations and tolerance must be positive')
    _, ix = spatial_partitions(shape)
    train = x[ix]
    ks = sorted(set(candidate_k))
    if not ks or any(not isinstance(k, (int, np.integer)) or not 1 <= k <= len(train) for k in ks):
        raise ValueError('GMM candidates must be integers in [1, fit sample size]')
    candidates, models = [], {}
    with threadpool_limits(limits=2):
        for k in ks:
            with warnings.catch_warnings(record=True) as caught:
                model = GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg_covar,
                                        n_init=n_init, max_iter=max_iter, tol=tol, random_state=seed).fit(train)
                if not model.converged_:
                    model.set_params(max_iter=max(600, max_iter * 2), warm_start=True, n_init=1)
                    model.fit(train)
            likelihood = float(model.score(train) * len(train))
            p = gmm_parameter_count(k, x.shape[1])
            score = -2 * likelihood + penalty * p * np.log(len(train))
            candidates.append({'k': k, 'log_likelihood': likelihood, 'parameter_count': p,
                               'criterion': float(score), 'converged': bool(model.converged_),
                               'warnings': [str(w.message) for w in caught]})
            models[k] = model
        valid = [c for c in candidates if c['converged']]
        if not valid:
            raise RuntimeError('No converged GMM candidate')
        chosen = min(valid, key=lambda c: (c['criterion'], c['k']))
        model = models[chosen['k']]
        labels = _predict(model, x, shape)
        confidence = np.empty(len(x), np.float32)
        for start in range(0, len(x), 65536):
            confidence[start:start + 65536] = model.predict_proba(x[start:start + 65536]).max(1)
    metadata = {'algorithm': 'gmm', 'covariance_type': 'full', 'penalty': penalty,
                'reg_covar': reg_covar, 'selected_k': chosen['k'], 'occupied_k': int(len(np.unique(labels))),
                'fit_sample_count': len(train), 'dimensions': x.shape[1], 'candidates': candidates,
                'at_search_boundary': chosen['k'] in [min(ks), max(ks)],
                'confidence_note': 'Model responsibility, not calibrated biological confidence'}
    return ClusterResult(labels, model, metadata, confidence.reshape(shape))
