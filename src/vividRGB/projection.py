"""Centered, unwhitened PCA and orthogonal varimax of projection weights."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAProjection:
    basis: np.ndarray
    mean: np.ndarray
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None

    def __post_init__(self):
        self.basis = np.asarray(self.basis, dtype=np.float32)
        self.mean = np.asarray(self.mean, dtype=np.float32)
        if self.basis.ndim != 2 or self.mean.shape != (self.basis.shape[0],):
            raise ValueError("PCA basis must have shape feature_dim x retained_dim, with matching mean")
        if not np.isfinite(self.basis).all() or not np.isfinite(self.mean).all():
            raise ValueError("PCA projection must be finite")

    def transform(self, tokens: np.ndarray, chunk_size: int = 8192):
        tokens = np.asarray(tokens)
        if tokens.shape[-1] != len(self.mean) or chunk_size < 1:
            raise ValueError("Token dimension does not match projection, or chunk size is invalid")
        flat = tokens.reshape(-1, tokens.shape[-1])
        projected = np.empty((len(flat), self.basis.shape[1]), np.float32)
        for start in range(0, len(flat), chunk_size):
            projected[start:start + chunk_size] = (flat[start:start + chunk_size].astype(np.float32) - self.mean) @ self.basis
        return projected.reshape(*tokens.shape[:-1], self.basis.shape[1])

    def save(self, path: str | Path):
        values = {"basis": self.basis, "mean": self.mean}
        for key in ["low", "high", "explained_variance_ratio"]:
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        np.savez_compressed(path, **values)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as z:
            return cls(**{k: z[k] for k in ["basis", "mean", "low", "high", "explained_variance_ratio"] if k in z})


def fit_pca(tokens: np.ndarray, n_components: int = 16, seed: int = 42):
    x = np.asarray(tokens, dtype=np.float32)
    if x.ndim != 2 or not 1 <= n_components <= min(x.shape) or not np.isfinite(x).all():
        raise ValueError("Require finite samples x features and an admissible PCA dimension")
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(x)
    scores = model.transform(x)
    low, high = np.quantile(scores[:, :min(3, n_components)], [.01, .99], axis=0)
    return PCAProjection(model.components_.T, model.mean_, low, high, model.explained_variance_ratio_)


def varimax(basis: np.ndarray, seed: int = 42, starts: int = 5, max_iter: int = 1000, tol: float = 1e-9):
    """Rotate all retained PCA axes; Kaiser normalization is used only in optimization.

    Returns the orthogonal rotation and diagnostics. Rotate original scores as X @ R.
    No whitening, extra variance weighting or feature L2 normalization is applied.
    """
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] < 1 or not np.isfinite(b).all():
        raise ValueError("Expected finite feature_dim x retained_dim basis")
    if starts < 1 or max_iter < 1 or tol <= 0:
        raise ValueError("Require positive starts, max_iter and tolerance")
    a = b / np.maximum(np.linalg.norm(b, axis=1)[:, None], 1e-15)
    p, k = a.shape
    rng, best = np.random.default_rng(seed), None
    def objective(r):
        l = a @ r
        return float(np.sum(l ** 4) - np.sum(np.sum(l * l, axis=0) ** 2) / p)
    for st in range(starts):
        r = np.eye(k) if st == 0 else np.linalg.qr(rng.normal(size=(k, k)))[0]
        before = objective(r)
        converged = False
        for it in range(max_iter):
            l = a @ r
            u, _, vh = np.linalg.svd(a.T @ (l ** 3 - l @ (np.diag(np.sum(l * l, axis=0)) / p)), full_matrices=False)
            new = u @ vh
            score = objective(new)
            if score + 1e-8 < before:
                raise RuntimeError("Varimax objective decreased")
            r = new
            if abs(score - before) <= tol * max(abs(before), 1.):
                converged = True
                break
            before = score
        result = {"rotation": r, "objective": objective(r), "iterations": it + 1,
                  "start": st, "converged": converged}
        if best is None or result["objective"] > best["objective"]:
            best = result
    r = best["rotation"]
    order = np.argsort(-np.sum((a @ r) ** 4, axis=0), kind="stable")
    r = r[:, order]
    weights = b @ r
    signs = np.where(weights[np.abs(weights).argmax(0), np.arange(k)] < 0, -1., 1.)
    best.update(rotation=r * signs, initial_objective=objective(np.eye(k)),
                kaiser_normalization=True, starts=starts)
    return best

