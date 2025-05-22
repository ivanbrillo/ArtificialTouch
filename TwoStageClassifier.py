import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from smoothingHelper import grid_max_smooth


class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_classifier=None, multiclass_classifier=None,
                 features=None,
                 binary_unbalanced=False, train_smoothed=None,
                 apply_smoothing=True):
        """
        Two-stage classifier that first determines if a sample is class 0 or non-zero (hard vs background points),
        then classifies non-zero samples into specific classes.

        Parameters:
        -----------
        binary_classifier : classifier object
            The classifier used for binary (zero vs non-zero) classification
        multiclass_classifier : classifier object
            The classifier used for multiclass classification among non-zero classes
        features : list or None
            Features to use for classification. If None, all features are used.
        binary_unbalanced : bool, default=False
            Whether to apply ADASYN resampling to handle unbalanced binary classes
        train_smoothed : list of dataframes
            Pre-smoothed training data for multiclass classification, in order to avoid recalculation
        apply_smoothing : bool, default=False
            Whether to apply grid max smoothing during prediction
        """
        self.binary_classifier = binary_classifier
        self.multiclass_classifier = multiclass_classifier
        self.features = features
        self.binary_unbalanced = binary_unbalanced
        self.train_smoothed = train_smoothed
        self.apply_smoothing = apply_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Binary task: classify as zero (y == 0) or non-zero (y > 0)
        X_binary = X[self.features] if self.features else X
        y_binary = (y > 0).astype(int)

        # Clone the binary classifier to make a proper estimator
        self.binary_model_ = clone(self.binary_classifier)

        if self.binary_unbalanced:  # Handle unbalanced binary classification with ADASYN
            adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=10)
            X_binary_ad, y_binary_ad = adasyn.fit_resample(X_binary, y_binary)
            self.binary_model_.fit(X_binary_ad, y_binary_ad)
        else:
            self.binary_model_.fit(X_binary, y_binary)

        # Multiclass task: classify among non-zero classes only
        mask_nonzero = y > 0

        if np.any(mask_nonzero):
            # Clone the multiclass classifier
            self.multiclass_model_ = clone(self.multiclass_classifier)
            y_nonzero = y[mask_nonzero] - 1

            if (X['fold'] == 0).all():
                X_nonzero = self.train_smoothed[0][self.features]
            elif (X['fold'] == 1).all():
                X_nonzero = self.train_smoothed[1][self.features]
            elif (X['fold'] == 2).all():
                X_nonzero = pd.concat([
                    self.train_smoothed[0][self.features],
                    self.train_smoothed[1][self.features]
                ], axis=0)
            else:
                raise Exception('Invalid fold identifiers or train_smoothed None')

            # Extract features for multiclass classifier
            X_nonzero_features = X_nonzero[self.features] if self.features else X_nonzero
            self.multiclass_model_.fit(X_nonzero_features, y_nonzero)

        return self

    def predict(self, X):

        # First, predict binary outcome
        X_binary = X[self.features] if self.features else X
        binary_pred = self.binary_model_.predict(X_binary)

        # Initialize predictions with zeros
        final_pred = np.zeros(len(X), dtype=int)

        # For samples predicted as non-zero, apply multiclass classifier
        nonzero_mask = binary_pred > 0

        if np.any(nonzero_mask):
            # Get the samples predicted as non-zero
            X_nonzero = X[nonzero_mask]

            # Apply smoothing if enabled
            if self.apply_smoothing:
                X_nonzero = grid_max_smooth(
                    X_nonzero,
                    features_to_smooth=self.features,
                )

            X_nonzero_features = X_nonzero[self.features] if self.features else X_nonzero
            multiclass_pred = self.multiclass_model_.predict(X_nonzero_features) + 1
            final_pred[nonzero_mask] = multiclass_pred

        return final_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        """
        # Get binary probabilities using the binary features
        X_binary = X[self.features] if self.features else X
        binary_proba = self.binary_model_.predict_proba(X_binary)

        # Get probability of being in class 0 vs non-zero
        prob_zero = binary_proba[:, 0]  # prob of being in class 0
        prob_nonzero = binary_proba[:, 1]  # prob of being in non-zero classes

        # Initialize final probability matrix with zeros
        final_proba = np.zeros((X.shape[0], len(self.classes_)))

        # Set the probability for class 0
        final_proba[:, 0] = prob_zero

        # Apply smoothing if enabled
        X_multi = X.copy()
        if self.apply_smoothing:
            X_multi = grid_max_smooth(
                X_multi,
                features_to_smooth=self.features,
            )

        # Extract features for multiclass classifier
        X_multi_features = X_multi[self.features] if self.features else X_multi

        # Get multiclass probabilities
        multiclass_proba = self.multiclass_model_.predict_proba(X_multi_features)

        # Scale the multiclass probabilities by the probability of being non-zero
        # For each non-zero class (1, 2, 3, etc.)
        for i in range(multiclass_proba.shape[1]):
            # The probability is: P(non-zero) * P(specific class | non-zero)
            final_proba[:, i + 1] = prob_nonzero * multiclass_proba[:, i]

        return final_proba
