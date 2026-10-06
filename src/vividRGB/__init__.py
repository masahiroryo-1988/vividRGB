"""Dense visual-cue discovery with frozen vision encoders."""
from .features import FocusConfig, FeatureResult, calibrate, extract_features, fourflip_encode
from .projection import PCAProjection, fit_pca, varimax
from .clustering import ClusterResult, cluster_kmeans, cluster_gmm
from .visualization import pca_rgb, labels_rgb

__version__ = "0.1.0"
__all__ = ["FocusConfig", "FeatureResult", "PCAProjection", "ClusterResult", "calibrate",
           "extract_features", "fourflip_encode", "fit_pca", "varimax", "cluster_kmeans",
           "cluster_gmm", "pca_rgb", "labels_rgb"]

