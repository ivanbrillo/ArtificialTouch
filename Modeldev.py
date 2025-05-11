import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Define the TwoStageClassifier class
class TwoStageClassifier:
    def __init__(self, binary_classifier, multiclass_classifier, binary_features=None, multiclass_features=None, binary_unbalanced=False):
        self.binary_classifier = binary_classifier
        self.multiclass_classifier = multiclass_classifier
        self.binary_features = binary_features
        self.multiclass_features = multiclass_features
        self.binary_unbalanced = binary_unbalanced
        if hasattr(multiclass_classifier, 'get_xgb_params') or type(multiclass_classifier).__name__.startswith('XGB'):
            self.is_xgb_multiclass = True
        else:
            self.is_xgb_multiclass = False

    def fit(self, X, y):
        # Binary task: classify as zero (y == 0) or non-zero (y > 0)
        X_binary = X[self.binary_features] if self.binary_features else X
        y_binary = (y > 0).astype(int)

        if self.binary_unbalanced:
            # Handle unbalanced binary classification
            adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=10)
            X_binary_ad, y_binary_ad = adasyn.fit_resample(X_binary, y_binary)
            self.binary_classifier.fit(X_binary_ad, y_binary_ad)
        else:
            self.binary_classifier.fit(X_binary, y_binary)

        # Multiclass task: classify among non-zero classes only
        mask_nonzero = y > 0
        if np.any(mask_nonzero):
            X_nonzero = X[mask_nonzero]
            y_nonzero = y[mask_nonzero]
            X_nonzero = X_nonzero[self.multiclass_features] if self.multiclass_features else X_nonzero
            self.multiclass_classifier.fit(X_nonzero, y_nonzero)

            # Only adjust labels for XGBoost
            if self.is_xgb_multiclass:
                # Store original class labels for prediction later
                self.original_classes = np.sort(np.unique(y_nonzero))
                # Shift labels to be zero-indexed
                y_nonzero = y_nonzero - 1

            self.multiclass_classifier.fit(X_nonzero, y_nonzero)

        return self

    def predict(self, X):
        # First, predict binary outcome
        X_binary = X[self.binary_features] if self.binary_features else X
        binary_pred = self.binary_classifier.predict(X_binary)

        # Initialize predictions with zeros
        final_pred = np.zeros(len(X))

        # For samples predicted as non-zero, apply multiclass classifier
        nonzero_mask = binary_pred > 0
        if np.any(nonzero_mask):
            X_nonzero = X[nonzero_mask]
            X_nonzero = X_nonzero[self.multiclass_features] if self.multiclass_features else X_nonzero
            multiclass_pred = self.multiclass_classifier.predict(X_nonzero)
            # If using XGBoost, shift predictions back to original scale
            if self.is_xgb_multiclass:
                multiclass_pred = multiclass_pred + 1
            final_pred[nonzero_mask] = multiclass_pred

        return final_pred

    def predict_proba(self, X):
        # Get binary probabilities using the binary features
        X_binary = X[self.binary_features] if self.binary_features else X
        binary_proba = self.binary_classifier.predict_proba(X_binary)

        # Get probability of being in class 0 vs non-zero
        prob_zero = binary_proba[:, 0]  # prob of being in class 0
        prob_nonzero = binary_proba[:, 1]  # prob of being in non-zero classes

        # Initialize final probability matrix with zeros
        # Shape: (num_samples, num_classes including 0)
        if self.is_xgb_multiclass:
            # For XGBoost, we need to account for the shifted labels
            num_classes = len(self.original_classes) + 1
        else:
            num_classes = len(np.unique(self.multiclass_classifier.classes_)) + 1

        final_proba = np.zeros((X.shape[0], num_classes))

        # Set the probability for class 0
        final_proba[:, 0] = prob_zero

        # Get multiclass probabilities for samples using the multiclass features
        # Note: We apply this to all samples but will scale by the probability of being non-zero
        X_multi = X[self.multiclass_features] if self.multiclass_features else X
        multiclass_proba = self.multiclass_classifier.predict_proba(X_multi)

        # Scale the multiclass probabilities by the probability of being non-zero
        # For each non-zero class (1, 2, 3, etc.)
        for i in range(multiclass_proba.shape[1]):
            # The probability is: P(non-zero) * P(specific class | non-zero)
            final_proba[:, i + 1] = prob_nonzero * multiclass_proba[:, i]

        return final_proba

def create_classifier(clf_type, params):
    if clf_type == 'rf':
        return RandomForestClassifier(**params, random_state=42)
    elif clf_type == 'svc':
        return SVC(**params, probability=True, random_state=42)
    elif clf_type == 'xgb':
        return xgb.XGBClassifier(**params, random_state=42)
    elif clf_type == 'logistic':
        return LogisticRegression(**params, random_state=42)
    else:  # 'gb'
        return GradientBoostingClassifier(**params, random_state=42)

# Define objective functions for Optuna hyperparameter tuning
def objective_rf(trial, X, y, groups, cv, ad = None):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }

    unique_groups = np.unique(groups)
    group_1_mask = groups == unique_groups[0]
    group_2_mask = groups == unique_groups[1]

    X_group1, y_group1 = X.iloc[group_1_mask], y[group_1_mask]
    X_group2, y_group2 = X.iloc[group_2_mask], y[group_2_mask]

    scores = []

    for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(
            cv.split(X_group1, y_group1, groups[group_1_mask]),
            cv.split(X_group2, y_group2, groups[group_2_mask])):

        # Scenario 1: Allena su gruppo1-train, valuta su gruppo2-val
        X_train = X_group1.iloc[train_idx1]
        y_train = y_group1[train_idx1]
        X_val = X_group2.iloc[val_idx2]
        y_val = y_group2[val_idx2]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = RandomForestClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

        # Scenario 2: Allena su gruppo2-train, valuta su gruppo1-val
        X_train = X_group2.iloc[train_idx2]
        y_train = y_group2[train_idx2]
        X_val = X_group1.iloc[val_idx1]
        y_val = y_group1[val_idx1]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = RandomForestClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

    return np.mean(scores)


def objective_svc(trial, X, y, groups, cv, ad = None):
    param = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 10, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear','poly']),
        'probability': True,
        'random_state': 42
    }
    if param['kernel'] == 'poly':
        param['degree'] = trial.suggest_int('degree', 2, 5)
    unique_groups = np.unique(groups)
    group_1_mask = groups == unique_groups[0]
    group_2_mask = groups == unique_groups[1]

    X_group1, y_group1 = X.iloc[group_1_mask], y[group_1_mask]
    X_group2, y_group2 = X.iloc[group_2_mask], y[group_2_mask]

    scores = []

    for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(
            cv.split(X_group1, y_group1, groups[group_1_mask]),
            cv.split(X_group2, y_group2, groups[group_2_mask])):

        X_train = X_group1.iloc[train_idx1]
        y_train = y_group1[train_idx1]
        X_val = X_group2.iloc[val_idx2]
        y_val = y_group2[val_idx2]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = SVC(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

        X_train = X_group2.iloc[train_idx2]
        y_train = y_group2[train_idx2]
        X_val = X_group1.iloc[val_idx1]
        y_val = y_group1[val_idx1]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = SVC(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

    return np.mean(scores)


def objective_xgb(trial, X, y, groups, cv, ad = None):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }

    unique_groups = np.unique(groups)
    group_1_mask = groups == unique_groups[0]
    group_2_mask = groups == unique_groups[1]

    X_group1, y_group1 = X[group_1_mask], y[group_1_mask]
    X_group2, y_group2 = X[group_2_mask], y[group_2_mask]

    scores = []

    for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(
            cv.split(X_group1, y_group1, groups[group_1_mask]),
            cv.split(X_group2, y_group2, groups[group_2_mask])):

        X_train = X_group1.iloc[train_idx1]
        y_train = y_group1[train_idx1]
        X_val = X_group2.iloc[val_idx2]
        y_val = y_group2[val_idx2]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = xgb.XGBClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

        X_train = X_group2.iloc[train_idx2]
        y_train = y_group2[train_idx2]
        X_val = X_group1.iloc[val_idx1]
        y_val = y_group1[val_idx1]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = xgb.XGBClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

    return np.mean(scores)


def objective_logistic(trial, X, y, groups, cv, ad = None):
    # Choose penalty type first
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', None, 'elasticnet'])

    # Define solver options based on penalty, but keep them as fixed lists in the code
    if penalty in ['l1', 'l2']:
        # Both liblinear and saga support these
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    elif penalty is None:
        # Only saga supports None penalty
        solver = 'saga'
    elif penalty == 'elasticnet':
        # Only saga supports elasticnet
        solver = 'saga'

    # Define the parameter dictionary
    param = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'penalty': penalty,
        'solver': solver,
        'max_iter': 1000,
        'random_state': 42
    }
    # Add l1_ratio only if penalty is elasticnet
    if penalty == 'elasticnet':
        param['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

    unique_groups = np.unique(groups)
    group_1_mask = groups == unique_groups[0]
    group_2_mask = groups == unique_groups[1]

    X_group1, y_group1 = X.iloc[group_1_mask], y[group_1_mask]
    X_group2, y_group2 = X.iloc[group_2_mask], y[group_2_mask]

    scores = []

    for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(
            cv.split(X_group1, y_group1, groups[group_1_mask]),
            cv.split(X_group2, y_group2, groups[group_2_mask])):

        X_train = X_group1.iloc[train_idx1]
        y_train = y_group1[train_idx1]
        X_val = X_group2.iloc[val_idx2]
        y_val = y_group2[val_idx2]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = LogisticRegression(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

        X_train = X_group2.iloc[train_idx2]
        y_train = y_group2[train_idx2]
        X_val = X_group1.iloc[val_idx1]
        y_val = y_group1[val_idx1]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = LogisticRegression(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

    return np.mean(scores)


def objective_gb(trial, X, y, groups, cv, ad = None):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42
    }

    unique_groups = np.unique(groups)
    group_1_mask = groups == unique_groups[0]
    group_2_mask = groups == unique_groups[1]

    X_group1, y_group1 = X.iloc[group_1_mask], y[group_1_mask]
    X_group2, y_group2 = X.iloc[group_2_mask], y[group_2_mask]

    scores = []

    for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(
            cv.split(X_group1, y_group1, groups[group_1_mask]),
            cv.split(X_group2, y_group2, groups[group_2_mask])):

        X_train = X_group1.iloc[train_idx1]
        y_train = y_group1[train_idx1]
        X_val = X_group2.iloc[val_idx2]
        y_val = y_group2[val_idx2]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = GradientBoostingClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

        X_train = X_group2.iloc[train_idx2]
        y_train = y_group2[train_idx2]
        X_val = X_group1.iloc[val_idx1]
        y_val = y_group1[val_idx1]

        if ad is not None:
            X_train, y_train = ad.fit_resample(X_train, y_train)

        clf = GradientBoostingClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

    return np.mean(scores)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate and return various classification metrics"""
    metrics = {}

    # Basic metrics
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')

    # Per class metrics
    classes = np.unique(np.concatenate([y_true, y_pred]))
    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        metrics[f'precision_class_{cls}'] = precision_score(y_true_bin, y_pred_bin)
        metrics[f'recall_class_{cls}'] = recall_score(y_true_bin, y_pred_bin)
        metrics[f'f1_class_{cls}'] = f1_score(y_true_bin, y_pred_bin)

    # AUC if probabilities are provided
    if y_proba is not None:
        if y_proba.shape[1] > 2:  # multiclass
            # One-vs-Rest ROC AUC
            metrics['auc_macro'] = roc_auc_score(
                pd.get_dummies(y_true), y_proba, average='macro', multi_class='ovr'
            )

            # Per class AUC
            for i, cls in enumerate(sorted(np.unique(y_true))):
                if i < y_proba.shape[1]:  # Ensure we have probabilities for this class
                    y_true_bin = (y_true == cls).astype(int)
                    y_proba_bin = y_proba[:, i]
                    try:
                        metrics[f'auc_class_{cls}'] = roc_auc_score(y_true_bin, y_proba_bin)
                    except:
                        metrics[f'auc_class_{cls}'] = np.nan
        else:  # binary
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])

    return metrics


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', normalize = True):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    if normalize:
        # Normalize by row (true label) to show the proportion of each class being predicted correctly/incorrectly
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # For classes with zero samples, replace NaN with zeros
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
        title = title + ' (Normalized)'
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add class distribution information
    class_distribution = np.sum(cm, axis=1)
    total_samples = np.sum(class_distribution)
    class_percentages = class_distribution / total_samples * 100

    plt.figtext(0.5, 0.01, f'Class Distribution: {class_distribution} samples ({class_percentages.round(1)}%)',
                ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout()
    plt.show()

    # Print additional metrics for unbalanced classification
    print(f"Class distribution: {class_distribution}")
    print(f"Class percentages: {class_percentages.round(1)}%")

    if normalize:
        # Print per-class accuracy (diagonal elements of normalized confusion matrix)
        per_class_accuracy = np.diag(cm_normalized)
        for i, acc in enumerate(per_class_accuracy):
            print(f"Class {i} accuracy: {acc:.2f}")


def plot_roc_curves(y_true, y_proba, classes):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 8))

    # Store FPR and TPR values for macro-averaging
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    # Plot individual class ROC curves
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        y_score = y_proba[:, i]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)

        plt.plot(fpr, tpr, lw=1.5, alpha=0.7,
                 label=f'Class {cls}')

        # Interpolate tpr values for macro-averaging
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    # Calculate and plot macro-averaged ROC curve
    mean_tpr /= len(classes)

    plt.plot(all_fpr, mean_tpr, 'b-', lw=2.5,
             label=f'Macro-average ROC')

    # Plot the diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

    # Configure the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (Per-Class and Macro Average)', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()