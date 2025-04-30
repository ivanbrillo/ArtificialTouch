# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, KFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
from typing import Dict, List, Tuple, Union, Optional, Callable
import warnings

warnings.filterwarnings('ignore')


class TwoStageClassifier:
    """Two-stage classifier for background vs materials classification."""

    def __init__(
            self,
            binary_classifier_pipeline: Union[Pipeline, ImbPipeline],
            multiclass_classifier_pipeline: Union[Pipeline, ImbPipeline]
    ):
        self.binary_classifier = binary_classifier_pipeline
        self.multiclass_classifier = multiclass_classifier_pipeline

    def fit(self, X, y):
        """Fit both stages of the classifier."""
        # Create binary labels for stage 1 (0: background, 1: material)
        binary_y = (y > 0).astype(int)

        # Fit binary classifier
        self.binary_classifier.fit(X, binary_y)

        # Filter data for material samples only for stage 2
        material_mask = y > 0
        if np.any(material_mask):
            X_materials = X[material_mask]
            y_materials = y[material_mask]

            # Fit multiclass classifier on material samples only
            self.multiclass_classifier.fit(X_materials, y_materials)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict using the two-stage approach."""
        # Stage 1: Binary classification
        binary_preds = self.binary_classifier.predict(X)

        # Initialize with background class
        final_preds = np.zeros(X.shape[0], dtype=int)

        # Stage 2: For samples predicted as materials, predict specific material
        material_mask = binary_preds > 0
        if np.any(material_mask):
            X_materials = X[material_mask]
            material_preds = self.multiclass_classifier.predict(X_materials)
            final_preds[material_mask] = material_preds

        return final_preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates for all classes."""

        # Get binary probabilities from stage 1
        binary_probs = self.binary_classifier.predict_proba(X)
        background_prob = binary_probs[:, 0].reshape(-1, 1)

        # Get material probabilities from stage 2 for all samples
        material_probs = self.multiclass_classifier.predict_proba(X)

        # Scale material probabilities by the probability of being material
        material_prob_scaling = binary_probs[:, 1].reshape(-1, 1)
        scaled_material_probs = material_probs * material_prob_scaling

        # Combine probabilities
        # First column is background probability, rest are scaled material probabilities
        combined_probs = np.hstack([background_prob, scaled_material_probs])

        return combined_probs


def create_pipeline(
        classifier: object,
        imputer_strategy: str = 'median',
        scaler_type: str = 'standard',
        smote_sampling: float = 1.0,
        n_features_to_select: Optional[int] = None
):
    """
    Create a scikit-learn pipeline with preprocessing, SMOTE(?), and classifier.
    """
    # Create imputer based on strategy
    imputer = SimpleImputer(strategy=imputer_strategy)

    # Create scaler based on type
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # Create feature selector (if needed)
    if n_features_to_select is not None and n_features_to_select > 0:
        # Start with estimator that has a feature_importances_ or coef_ attribute
        if (hasattr(classifier, 'feature_importances_') or hasattr(classifier, 'coef_')) and hasattr(classifier,
                                                                                                     'class_weight'):
            selector_model = clone(classifier)
            selector_model.class_weight = "balanced"
            selector = RFE(estimator=selector_model, n_features_to_select=n_features_to_select)
            pipeline_steps = [
                ('imputer', imputer),
                ('scaler', scaler),
                ('selector', selector),
                ('smote', SMOTE(sampling_strategy=smote_sampling, random_state=42)),
                ('classifier', classifier)
            ]
        else:
            # For models without feature importance attributes
            temp_estimator = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
            selector = RFE(estimator=temp_estimator, n_features_to_select=n_features_to_select)
            pipeline_steps = [
                ('imputer', imputer),
                ('scaler', scaler),
                ('selector', selector),
                ('smote', SMOTE(sampling_strategy=smote_sampling, random_state=42)),
                ('classifier', classifier)
            ]
    else:
        pipeline_steps = [
            ('imputer', imputer),
            ('scaler', scaler),
            ('smote', SMOTE(sampling_strategy=smote_sampling, random_state=42)),
            ('classifier', classifier)
        ]

    return ImbPipeline(steps=pipeline_steps)


def get_base_classifiers() -> Dict[str, object]:
    """Get dictionary of base classifier objects."""
    return {
        'rf': RandomForestClassifier(random_state=42),
        'svc': SVC(probability=True, random_state=42),
        'xgb': xgb.XGBClassifier(objective='multi:softprob', random_state=42),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'gb': GradientBoostingClassifier(random_state=42)
    }


def define_hyperparameter_space(classifier_name: str) -> Dict:
    """
    Define hyperparameter space for Optuna optimization based on classifier type.
    """
    # Common preprocessing parameters
    common_params = {
        'imputer_strategy': ['mean', 'median', 'most_frequent'],
        'scaler_type': ['standard', 'robust'],
        'smote_sampling': [0.8, 0.9, 1.0, 1.1],
    }

    # Classifier-specific parameters
    if classifier_name == 'rf':
        clf_params = {
            'n_estimators': (50, 300),
            'max_depth': (3, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None],
        }
    elif classifier_name == 'svc':
        clf_params = {
            'C': (0.1, 10.0, 'log'),
            'gamma': (0.001, 1.0, 'log'),
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': (2, 5),
        }
    elif classifier_name == 'xgb':
        clf_params = {
            'n_estimators': (50, 300),
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3, 'log'),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'min_child_weight': (1, 10),
        }
    elif classifier_name == 'logistic':
        clf_params = {
            'C': (0.01, 10.0, 'log'),
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
            'l1_ratio': (0.0, 1.0),
        }
    elif classifier_name == 'gb':
        clf_params = {
            'n_estimators': (50, 300),
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3, 'log'),
            'subsample': (0.5, 1.0),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
        }
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return {**common_params, **clf_params, 'n_features_to_select_ratio': (0.2, 1.0)}


def objective(
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        classifier_name: str,
        metric: str = 'f1_macro'
):
    """
    Objective function for Optuna optimization.
    """
    base_classifiers = get_base_classifiers()
    classifier = base_classifiers[classifier_name]

    # Get hyperparameter space
    param_space = define_hyperparameter_space(classifier_name)

    # Sample hyperparameters
    imputer_strategy = trial.suggest_categorical(
        'imputer_strategy', param_space['imputer_strategy']
    )
    scaler_type = trial.suggest_categorical(
        'scaler_type', param_space['scaler_type']
    )
    smote_sampling = trial.suggest_categorical(
        'smote_sampling', param_space['smote_sampling']
    )

    # Feature selection ratio (proportion of features to keep)
    n_features_ratio = trial.suggest_float(
        'n_features_to_select_ratio',
        param_space['n_features_to_select_ratio'][0],
        param_space['n_features_to_select_ratio'][1]
    )
    n_features_to_select = max(1, int(X.shape[1] * n_features_ratio))

    # Set classifier-specific hyperparameters
    if classifier_name == 'rf':
        classifier.n_estimators = trial.suggest_int(
            'n_estimators', param_space['n_estimators'][0], param_space['n_estimators'][1]
        )
        classifier.max_depth = trial.suggest_int(
            'max_depth', param_space['max_depth'][0], param_space['max_depth'][1]
        )
        classifier.min_samples_split = trial.suggest_int(
            'min_samples_split', param_space['min_samples_split'][0], param_space['min_samples_split'][1]
        )
        classifier.min_samples_leaf = trial.suggest_int(
            'min_samples_leaf', param_space['min_samples_leaf'][0], param_space['min_samples_leaf'][1]
        )
        classifier.max_features = trial.suggest_categorical(
            'max_features', param_space['max_features']
        )

    elif classifier_name == 'svc':
        classifier.C = trial.suggest_float(
            'C', param_space['C'][0], param_space['C'][1], log=True
        )
        classifier.gamma = trial.suggest_float(
            'gamma', param_space['gamma'][0], param_space['gamma'][1], log=True
        )
        classifier.kernel = trial.suggest_categorical(
            'kernel', param_space['kernel']
        )
        if classifier.kernel == 'poly':
            classifier.degree = trial.suggest_int(
                'degree', param_space['degree'][0], param_space['degree'][1]
            )

    elif classifier_name == 'xgb':
        classifier.n_estimators = trial.suggest_int(
            'n_estimators', param_space['n_estimators'][0], param_space['n_estimators'][1]
        )
        classifier.max_depth = trial.suggest_int(
            'max_depth', param_space['max_depth'][0], param_space['max_depth'][1]
        )
        classifier.learning_rate = trial.suggest_float(
            'learning_rate', param_space['learning_rate'][0], param_space['learning_rate'][1], log=True
        )
        classifier.subsample = trial.suggest_float(
            'subsample', param_space['subsample'][0], param_space['subsample'][1]
        )
        classifier.colsample_bytree = trial.suggest_float(
            'colsample_bytree', param_space['colsample_bytree'][0], param_space['colsample_bytree'][1]
        )
        classifier.min_child_weight = trial.suggest_int(
            'min_child_weight', param_space['min_child_weight'][0], param_space['min_child_weight'][1]
        )

    elif classifier_name == 'logistic':
        classifier.C = trial.suggest_float(
            'C', param_space['C'][0], param_space['C'][1], log=True
        )
        classifier.penalty = trial.suggest_categorical(
            'penalty', param_space['penalty']
        )
        # Handle solver constraints
        if classifier.penalty == 'elasticnet':
            classifier.solver = 'saga'
            classifier.l1_ratio = trial.suggest_float(
                'l1_ratio', param_space['l1_ratio'][0], param_space['l1_ratio'][1]
            )
        elif classifier.penalty == 'l1':
            classifier.solver = 'liblinear'
        else:
            classifier.solver = trial.suggest_categorical(
                'solver', param_space['solver']
            )

    elif classifier_name == 'gb':
        classifier.n_estimators = trial.suggest_int(
            'n_estimators', param_space['n_estimators'][0], param_space['n_estimators'][1]
        )
        classifier.max_depth = trial.suggest_int(
            'max_depth', param_space['max_depth'][0], param_space['max_depth'][1]
        )
        classifier.learning_rate = trial.suggest_float(
            'learning_rate', param_space['learning_rate'][0], param_space['learning_rate'][1], log=True
        )
        classifier.subsample = trial.suggest_float(
            'subsample', param_space['subsample'][0], param_space['subsample'][1]
        )
        classifier.min_samples_split = trial.suggest_int(
            'min_samples_split', param_space['min_samples_split'][0], param_space['min_samples_split'][1]
        )
        classifier.min_samples_leaf = trial.suggest_int(
            'min_samples_leaf', param_space['min_samples_leaf'][0], param_space['min_samples_leaf'][1]
        )

    # Create pipeline with the optimized parameters
    pipeline = create_pipeline(
        classifier=classifier,
        imputer_strategy=imputer_strategy,
        scaler_type=scaler_type,
        smote_sampling=smote_sampling,
        n_features_to_select=n_features_to_select
    )

    # Set up cross-validation
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True)
        cv_splits = cv.split(X, y, groups)
    else:
        raise ValueError('Groups must be defined for Group cross-validation.')

    # Define scoring metrics
    scoring = {
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'roc_auc_ovr': 'roc_auc_ovr_weighted'
    }

    # Perform cross-validation
    try:
        cv_results = cross_validate(
            pipeline, X, y,
            cv=cv_splits, scoring=scoring,
            return_estimator=False
        )

        # Return the mean score for the target metric
        return cv_results[f'test_{metric}'].mean()

    except Exception as e:
        print(f"Error in CV: {e}")
        return 0.0


def optimize_pipeline(
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        classifier_name: str,
        n_trials: int = 30,
        metric: str = 'f1_macro'
):
    """
    Optimize pipeline hyperparameters using Optuna.
    """
    study = optuna.create_study(direction='maximize')

    # Create partial function with fixed arguments
    objective_func = lambda trial: objective(
        trial, X, y, groups, classifier_name, metric
    )

    # Run optimization
    study.optimize(objective_func, n_trials=n_trials, n_jobs=-1)

    return study.best_params, study.best_value


def create_optimized_pipeline(
        classifier_name: str,
        best_params: Dict,
        X: pd.DataFrame
):
    """
    Create pipeline with optimized hyperparameters.
    """
    base_classifiers = get_base_classifiers()
    classifier = base_classifiers[classifier_name]

    # Set classifier-specific hyperparameters
    if classifier_name == 'rf':
        classifier.n_estimators = best_params.get('n_estimators', 100)
        classifier.max_depth = best_params.get('max_depth', None)
        classifier.min_samples_split = best_params.get('min_samples_split', 2)
        classifier.min_samples_leaf = best_params.get('min_samples_leaf', 1)
        classifier.max_features = best_params.get('max_features', 'sqrt')

    elif classifier_name == 'svc':
        classifier.C = best_params.get('C', 1.0)
        classifier.gamma = best_params.get('gamma', 'scale')
        classifier.kernel = best_params.get('kernel', 'rbf')
        if classifier.kernel == 'poly':
            classifier.degree = best_params.get('degree', 3)

    elif classifier_name == 'xgb':
        classifier.n_estimators = best_params.get('n_estimators', 100)
        classifier.max_depth = best_params.get('max_depth', 6)
        classifier.learning_rate = best_params.get('learning_rate', 0.1)
        classifier.subsample = best_params.get('subsample', 1.0)
        classifier.colsample_bytree = best_params.get('colsample_bytree', 1.0)
        classifier.min_child_weight = best_params.get('min_child_weight', 1)

    elif classifier_name == 'logistic':
        classifier.C = best_params.get('C', 1.0)
        classifier.penalty = best_params.get('penalty', 'l2')
        classifier.solver = best_params.get('solver', 'lbfgs')
        if classifier.penalty == 'elasticnet':
            classifier.l1_ratio = best_params.get('l1_ratio', 0.5)

    elif classifier_name == 'gb':
        classifier.n_estimators = best_params.get('n_estimators', 100)
        classifier.max_depth = best_params.get('max_depth', 3)
        classifier.learning_rate = best_params.get('learning_rate', 0.1)
        classifier.subsample = best_params.get('subsample', 1.0)
        classifier.min_samples_split = best_params.get('min_samples_split', 2)
        classifier.min_samples_leaf = best_params.get('min_samples_leaf', 1)

    # Calculate number of features to select
    n_features_ratio = best_params.get('n_features_to_select_ratio', 1.0)
    n_features_to_select = max(1, int(X.shape[1] * n_features_ratio))

    # Create pipeline with optimized parameters
    return create_pipeline(
        classifier=classifier,
        imputer_strategy=best_params.get('imputer_strategy', 'median'),
        scaler_type=best_params.get('scaler_type', 'standard'),
        smote_sampling=best_params.get('smote_sampling', 1.0),
        n_features_to_select=n_features_to_select
    )


def evaluate_classifier(
        classifier,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        prefix: str = ''
):
    """
    Evaluate classifier with cross-validation.
    """
    # Set up cross-validation
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True)
        cv_splits = cv.split(X, y, groups)
    else:
        raise ValueError('Groups must be defined for Group cross-validation.')

    # Define scoring metrics
    scoring = {
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'roc_auc_ovr': 'roc_auc_ovr_weighted'
    }

    # Perform cross-validation
    cv_results = cross_validate(
        classifier, X, y,
        cv=cv_splits, scoring=scoring,
        return_estimator=True
    )

    # Aggregate results
    results = {}
    for metric in scoring.keys():
        results[f'{prefix}{metric}'] = cv_results[f'test_{metric}'].mean()
        results[f'{prefix}{metric}_std'] = cv_results[f'test_{metric}'].std()

    # Store cross-validation estimators for later use (e.g., ROC curves)
    results['cv_estimators'] = cv_results['estimator']
    results['cv_scores'] = {metric: cv_results[f'test_{metric}'] for metric in scoring.keys()}

    return results


def nested_cross_validation(
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        n_outer_splits: int = 8,
        n_trials: int = 20,
        classifier_names: List[str] = None,
        metric: str = 'f1_macro'
):
    """
    Perform nested cross-validation for model selection and evaluation.

    Args:
        X: Features
        y: Target
        groups: Group identifiers
        n_outer_splits: Number of outer CV splits
        n_trials: Number of Optuna trials per inner CV
        classifier_names: List of classifier names to evaluate
        metric: Metric to optimize

    Returns:
        Dictionary with results for each classifier
    """
    if classifier_names is None:
        classifier_names = ['rf', 'svc', 'xgb', 'logistic', 'gb']

        # Set up outer cross-validation
    if groups is not None:
        outer_cv = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True)
    else:
        raise ValueError('Groups must be defined for Group cross-validation.')

        # Dictionary to store results
    all_results = {
        'stage1': {name: {'scores': [], 'params': []} for name in classifier_names},
        'stage2': {name: {'scores': [], 'params': []} for name in classifier_names}
    }

    # Run nested cross-validation
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y, groups):
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]

        # Create binary labels for stage 1
        y_binary_train = (y_outer_train > 0).astype(int)
        y_binary_test = (y_outer_test > 0).astype(int)

        # Extract material samples for stage 2
        material_mask_train = y_outer_train > 0
        X_material_train = X_outer_train[material_mask_train]
        y_material_train = y_outer_train[material_mask_train]
        groups_material_train = groups_outer_train[material_mask_train] if groups_outer_train is not None else None

        material_mask_test = y_outer_test > 0
        X_material_test = X_outer_test[material_mask_test]
        y_material_test = y_outer_test[material_mask_test]

        # For each classifier type
        for clf_name in classifier_names:
            print(f"\nTraining {clf_name} in outer CV fold...")

            # Stage 1: Binary classifier optimization with inner CV
            print(f"Optimizing Stage 1 (binary) classifier...")
            stage1_best_params, stage1_best_score = optimize_pipeline(
                X_outer_train, y_binary_train, groups_outer_train,
                clf_name, n_trials, metric
            )

            # Create Stage 1 pipeline with optimized parameters
            stage1_pipeline = create_optimized_pipeline(
                clf_name, stage1_best_params, X_outer_train
            )

            # Fit and evaluate Stage 1 independently
            stage1_pipeline.fit(X_outer_train, y_binary_train)
            stage1_preds = stage1_pipeline.predict(X_outer_test)

            # Calculate Stage 1 metrics
            stage1_fold_scores = {
                'f1_macro': f1_score(y_binary_test, stage1_preds, average='macro'),
                'precision_macro': precision_score(y_binary_test, stage1_preds, average='macro'),
                'recall_macro': recall_score(y_binary_test, stage1_preds, average='macro')
            }

            # Calculate ROC-AUC for Stage 1 if possible
            try:
                stage1_pred_proba = stage1_pipeline.predict_proba(X_outer_test)
                stage1_fold_scores['roc_auc_macro'] = roc_auc_score(
                    y_binary_test, stage1_pred_proba[:, 1], average='macro'
                )
            except Exception as e:
                print(f"Could not compute ROC-AUC for Stage 1: {e}")
                stage1_fold_scores['roc_auc_macro'] = np.nan

            # Store Stage 1 results
            all_results['stage1'][clf_name]['scores'].append(stage1_fold_scores)
            all_results['stage1'][clf_name]['params'].append(stage1_best_params)

            # Stage 2: Multiclass classifier optimization with inner CV (only if we have material samples)
            if len(X_material_train) > 0:
                print(f"Optimizing Stage 2 (multiclass) classifier...")
                stage2_best_params, stage2_best_score = optimize_pipeline(
                    X_material_train, y_material_train, groups_material_train,
                    clf_name, n_trials, metric
                )

                # Create Stage 2 pipeline with optimized parameters
                stage2_pipeline = create_optimized_pipeline(
                    clf_name, stage2_best_params, X_material_train
                )

                # Fit and evaluate Stage 2 independently (only on material samples)
                if len(X_material_test) > 0:
                    stage2_pipeline.fit(X_material_train, y_material_train)
                    stage2_preds = stage2_pipeline.predict(X_material_test)

                    # Calculate Stage 2 metrics
                    stage2_fold_scores = {
                        'f1_macro': f1_score(y_material_test, stage2_preds, average='macro'),
                        'precision_macro': precision_score(y_material_test, stage2_preds, average='macro'),
                        'recall_macro': recall_score(y_material_test, stage2_preds, average='macro')
                    }

                    # Calculate ROC-AUC for Stage 2 if possible
                    try:
                        stage2_pred_proba = stage2_pipeline.predict_proba(X_material_test)
                        stage2_fold_scores['roc_auc_macro'] = roc_auc_score(
                            y_material_test, stage2_pred_proba, multi_class='ovr', average='macro'
                        )
                    except Exception as e:
                        print(f"Could not compute ROC-AUC for Stage 2: {e}")
                        stage2_fold_scores['roc_auc_macro'] = np.nan

                    # Store Stage 2 results
                    all_results['stage2'][clf_name]['scores'].append(stage2_fold_scores)
                    all_results['stage2'][clf_name]['params'].append(stage2_best_params)
                else:
                    print("No material samples in test set for Stage 2 evaluation")
            else:
                print("No material samples in training set for Stage 2 optimization")

    # Calculate aggregate metrics for each stage
    for stage in ['stage1', 'stage2']:
        for clf_name in classifier_names:
            all_results[stage][clf_name]['mean_scores'] = {}
            all_results[stage][clf_name]['std_scores'] = {}

            # Skip if no scores for this stage/classifier
            if not all_results[stage][clf_name]['scores']:
                continue

            for metric in ['f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']:
                metric_values = [s.get(metric, np.nan) for s in all_results[stage][clf_name]['scores']
                                 if metric in s and not np.isnan(s[metric])]

                if metric_values:
                    all_results[stage][clf_name]['mean_scores'][metric] = np.mean(metric_values)
                    all_results[stage][clf_name]['std_scores'][metric] = np.std(metric_values)
                else:
                    all_results[stage][clf_name]['mean_scores'][metric] = np.nan
                    all_results[stage][clf_name]['std_scores'][metric] = np.nan

    return all_results


def get_best_classifier(
        cv_results: Dict,
        metric: str = 'f1_macro'
):
    """
    Select the best classifier for each stage based on nested CV results.
    """
    best_classifiers = {}

    for stage in ['stage1', 'stage2']:
        best_score = -np.inf
        best_clf_name = None
        best_params = None

        for clf_name, results in cv_results[stage].items():
            if 'mean_scores' not in results:
                continue

            mean_score = results['mean_scores'].get(metric, -np.inf)
            if mean_score > best_score and not np.isnan(mean_score):
                best_score = mean_score
                best_clf_name = clf_name

                # Get best parameters across folds
                scores = [s.get(metric, -np.inf) for s in results['scores'] if metric in s]
                if scores:
                    best_idx = np.argmax(scores)
                    best_params = results['params'][best_idx]

        best_classifiers[stage] = {
            'classifier_name': best_clf_name,
            'params': best_params,
            'score': best_score
        }

    return best_classifiers


def train_final_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_classifiers: Dict
) -> Tuple[TwoStageClassifier, Dict[str, float]]:
    best_clf_name_1 = best_classifiers['stage1']['classifier_name']
    best_params_1 = best_classifiers['stage1']['params']

    # Create stage 1 pipeline (binary classifier)
    stage1_pipeline = create_optimized_pipeline(
        best_clf_name_1, best_params_1, X_train
    )

    best_clf_name_2 = best_classifiers['stage2']['classifier_name']
    best_params_2 = best_classifiers['stage2']['params']

    # Create stage 2 pipeline (multiclass classifier)
    material_mask = y_train > 0
    X_material_train = X_train[material_mask]
    stage2_pipeline = create_optimized_pipeline(
        best_clf_name_2, best_params_2, X_material_train
    )

    # Create and train two-stage classifier
    final_model = TwoStageClassifier(
        binary_classifier_pipeline=stage1_pipeline,
        multiclass_classifier_pipeline=stage2_pipeline
    )
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = final_model.predict(X_test)

    # Calculate metrics
    test_metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Get probability predictions for ROC curve calculation
    y_pred_proba = final_model.predict_proba(X_test)

    # Get unique classes present in the dataset
    classes = np.unique(np.concatenate([y_train.unique(), y_test.unique()]))
    n_classes = len(classes)

    # Calculate and store ROC curve data for plotting
    try:
        # Overall ROC AUC score
        test_metrics['roc_auc_macro'] = roc_auc_score(
            y_test, y_pred_proba, multi_class='ovr', average='macro'
        )

        # Create binary indicator matrix for ROC curve calculation
        y_test_bin = np.zeros((len(y_test), n_classes))
        for i, c in enumerate(classes):
            y_test_bin[:, i] = (y_test == c).astype(int)

        # Compute ROC curves for each class (one-vs-rest)
        roc_curves = {'class_curves': {}}
        for i, class_idx in enumerate(classes):
            # Skip if the class is not present in test set
            if np.sum(y_test == class_idx) == 0:
                continue

            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            roc_curves['class_curves'][class_idx] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }

        # Calculate macro-average ROC curve
        all_fpr = np.unique(np.concatenate([curve_data['fpr'] for curve_data in roc_curves['class_curves'].values()]))
        mean_tpr = np.zeros_like(all_fpr)

        for class_idx, curve_data in roc_curves['class_curves'].items():
            mean_tpr += np.interp(all_fpr, curve_data['fpr'], curve_data['tpr'])

        mean_tpr /= len(roc_curves['class_curves'])
        macro_auc = auc(all_fpr, mean_tpr)

        roc_curves['macro'] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': macro_auc
        }

        # Store ROC curve data in metrics
        test_metrics['roc_curves'] = roc_curves

    except Exception as e:
        print(f"Could not compute ROC curves: {e}")
        test_metrics['roc_auc_macro'] = np.nan

    # Add predicted probabilities and true labels for additional analysis if needed
    test_metrics['y_true'] = y_test
    test_metrics['y_pred'] = y_pred
    test_metrics['y_pred_proba'] = y_pred_proba
    test_metrics['classes'] = classes

    return final_model, test_metrics


def plot_evaluation_metrics(
        cv_results: Dict[str, Dict],
        metric_names: List[str] = None
) -> None:
    """
    Plot boxplots comparing evaluation metrics across models.

    Args:
        cv_results: Results from nested_cross_validation
        metric_names: List of metrics to plot
    """
    if metric_names is None:
        metric_names = ['f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']

    classifier_names = list(cv_results.keys())

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]

    for i, metric in enumerate(metric_names):
        # Collect scores across classifiers
        all_scores = []
        for clf_name in classifier_names:
            scores = [s.get(metric, np.nan) for s in cv_results[clf_name]['scores']]
            all_scores.append(scores)

        # Create boxplot
        axes[i].boxplot(all_scores, labels=classifier_names)
        axes[i].set_title(f'{metric} across classifiers')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
        test_metrics: Dict,
        class_names: List[str] = None
) -> None:
    """
    Plot confusion matrix from test metrics.

    Args:
        test_metrics: Dictionary containing evaluation metrics including confusion matrix
        class_names: Names of classes
    """
    # Extract confusion matrix from test metrics
    cm = test_metrics['confusion_matrix']

    # Get number of classes from confusion matrix
    n_classes = cm.shape[0]

    # If class names not provided, use indices as names
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(
        test_metrics: Dict,
        class_names: List[str] = None
) -> None:
    """
    Plot ROC curves using pre-computed roc curves from test metrics.

    Args:
        test_metrics: Dictionary containing evaluation metrics including ROC curve data
        class_names: Names of classes
    """
    # Check if ROC AUC was successfully computed
    if 'roc_auc_macro' not in test_metrics or np.isnan(test_metrics['roc_auc_macro']):
        print("ROC AUC not available in test metrics")
        return

    # Check if we have the ROC curve data
    if 'roc_curves' not in test_metrics:
        print("ROC curve data not found in test metrics. Make sure to save it during evaluation.")
        return

    roc_data = test_metrics['roc_curves']

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot ROC curves for each class
    for i, (class_idx, curve_data) in enumerate(roc_data['class_curves'].items()):
        fpr = curve_data['fpr']
        tpr = curve_data['tpr']
        roc_auc = curve_data['auc']

        # Use class name if provided, otherwise use index
        class_label = class_names[i] if class_names and i < len(class_names) else f'Class {class_idx}'

        # Plot class ROC curve
        plt.plot(
            fpr, tpr, lw=2,
            label=f'{class_label} (AUC = {roc_auc:.2f})'
        )

    # Plot macro-average ROC curve if available
    if 'macro' in roc_data:
        macro_fpr = roc_data['macro']['fpr']
        macro_tpr = roc_data['macro']['tpr']
        macro_auc = test_metrics['roc_auc_macro']

        plt.plot(
            macro_fpr, macro_tpr, lw=2, linestyle='--', color='crimson',
            label=f'Macro-average (AUC = {macro_auc:.2f})'
        )

    # Overall AUC from test metrics
    overall_auc = test_metrics['roc_auc_macro']

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Overall AUC: {overall_auc:.4f}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
