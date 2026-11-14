"""The reduction of one band, fitted on training rows and applied to any rows."""

import numpy as np


class FeatureProjection:
    """Truncated SVD to at most ``feature_cap`` components.

    A band with ``feature_cap`` units or fewer is kept as it is. ``transform``
    refuses features of another width or another unit order than the fit.
    """

    def __init__(self, feature_cap):
        self.feature_cap = feature_cap
        self.n_features_in_ = None
        self.feature_ids_ = None
        self.svd_ = None

    def fit(self, features, feature_ids=None):
        self.n_features_in_ = features.shape[1]
        self.feature_ids_ = None if feature_ids is None else np.array(feature_ids, copy=True)
        self.svd_ = None
        if self.n_features_in_ > self.feature_cap:
            from sklearn.decomposition import TruncatedSVD
            self.svd_ = TruncatedSVD(
                n_components=min(self.feature_cap, features.shape[0]), random_state=0)
            self.svd_.fit(features)
        return self

    def transform(self, features, feature_ids=None):
        if self.n_features_in_ is None:
            raise ValueError('Fit the feature projection on training data before prediction.')
        if features.shape[1] != self.n_features_in_:
            raise ValueError('Prediction feature width differs from the training feature basis.')
        if self.feature_ids_ is not None and (
                feature_ids is None or not np.array_equal(self.feature_ids_, feature_ids)):
            raise ValueError('Prediction feature identities/order differ from the training feature basis.')
        return (self.svd_.transform(features) if self.svd_ is not None
                else np.asarray(features, dtype=np.float64))
