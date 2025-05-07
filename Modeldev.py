import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')


# Define the TwoStageClassifier class
class TwoStageClassifier:
    def __init__(self, binary_classifier, multiclass_classifier):
        self.binary_classifier = binary_classifier
        self.multiclass_classifier = multiclass_classifier

    def fit(self, X, y):
        # Binary task: classify as zero (y == 0) or non-zero (y > 0)
        y_binary = (y > 0).astype(int)
        self.binary_classifier.fit(X, y_binary)

        # Multiclass task: classify among non-zero classes only
        mask_nonzero = y > 0
        if np.any(mask_nonzero):
            X_nonzero = X[mask_nonzero]
            y_nonzero = y[mask_nonzero]
            self.multiclass_classifier.fit(X_nonzero, y_nonzero)

        return self

    def predict(self, X):
        # First, predict binary outcome
        binary_pred = self.binary_classifier.predict(X)

        # Initialize predictions with zeros
        final_pred = np.zeros(len(X))

        # For samples predicted as non-zero, apply multiclass classifier
        nonzero_mask = binary_pred > 0
        if np.any(nonzero_mask):
            X_nonzero = X[nonzero_mask]
            multiclass_pred = self.multiclass_classifier.predict(X_nonzero)
            final_pred[nonzero_mask] = multiclass_pred

        return final_pred

    def predict_proba(self, X):
        # Get binary probabilities
        binary_proba = self.binary_classifier.predict_proba(X)

        # Get probability of being in class 0
        prob_class0 = binary_proba[:, 0:1]  # prob of being in class 0

        # Initialize final probability matrix with zeros
        # (Number of samples x Number of classes including 0)
        num_classes = len(np.unique(self.multiclass_classifier.classes_)) + 1
        final_proba = np.zeros((X.shape[0], num_classes))

        # Set the probability for class 0
        final_proba[:, 0] = prob_class0.ravel()

        # Get multiclass probabilities for all samples
        multiclass_proba = self.multiclass_classifier.predict_proba(X)

        # Scale multiclass probabilities by probability of being non-zero
        for i in range(multiclass_proba.shape[1]):
            final_proba[:, i + 1] = binary_proba[:, 1] * multiclass_proba[:, i]

        return final_proba


# Define objective functions for Optuna hyperparameter tuning
def objective_rf(trial, X, y, groups, cv):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }

    clf = RandomForestClassifier(**param)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_pred, average='macro'))

    return np.mean(scores)


def objective_svc(trial, X, y, groups, cv):
    param = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
        'probability': True,
        'random_state': 42
    }

    clf = SVC(**param)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_pred, average='macro'))

    return np.mean(scores)


def objective_xgb(trial, X, y, groups, cv):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }

    clf = xgb.XGBClassifier(**param)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_pred, average='macro'))

    return np.mean(scores)


def objective_logistic(trial, X, y, groups, cv):
    param = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'max_iter': 1000,
        'random_state': 42
    }

    if len(np.unique(y)) > 2:
        param['multi_class'] = 'multinomial'
        if param['solver'] == 'liblinear':
            param['solver'] = 'saga'  # liblinear doesn't support multinomial

    clf = LogisticRegression(**param)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_pred, average='macro'))

    return np.mean(scores)


def objective_gb(trial, X, y, groups, cv):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42
    }

    clf = GradientBoostingClassifier(**param)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_pred, average='macro'))

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


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curves(y_true, y_proba, classes):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 8))

    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        y_score = y_proba[:, i]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc_score(y_true_bin, y_score):.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

