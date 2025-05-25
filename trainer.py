import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from TwoStageClassifier import TwoStageClassifier
from plotter import plot_confusion_matrix


def fit_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []

    for i, (name, model) in enumerate(models):
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluation metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)

        print(f"{name} Training Accuracy: {train_acc:.4f}")
        print(f"{name} Training Balanced Accuracy: {train_bal_acc:.4f}")
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        print(f"{name} Test Balanced Accuracy: {test_bal_acc:.4f}")

        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Train Balanced Accuracy': train_bal_acc,
            'Test Accuracy': test_acc,
            'Test Balanced Accuracy': test_bal_acc
        })

        labels = list(set(np.unique(y_train)).union(set(np.unique(y_test))))

        # Plot for this model
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_confusion_matrix(y_train, y_train_pred, f'{name} - Train', axes[0], class_labels=labels)
        plot_confusion_matrix(y_test, y_test_pred, f'{name} - Test', axes[1], class_labels=labels)
        plt.tight_layout()
        plt.show()

    return results


def get_best_model(random_search, hard_spots_train_smoothed, hard_spots_validation_smoothed):
    # Get the best parameters
    best_params = random_search.best_params_

    # Create base estimators
    binary_model = clone(best_params['binary_classifier'])
    multiclass_model = clone(best_params['multiclass_classifier'])
    best_features = best_params['features'] + ["posx", "posy"]

    # Apply parameters to the two models
    for param, value in best_params.items():
        if param.startswith('binary_classifier__'):
            param_name = param.replace('binary_classifier__', '')
            setattr(binary_model, param_name, value)
        if param.startswith('multiclass_classifier__'):
            param_name = param.replace('multiclass_classifier__', '')
            setattr(multiclass_model, param_name, value)

    # Create classifier with the configured models
    best_clf = TwoStageClassifier(
        binary_classifier=binary_model,
        multiclass_classifier=multiclass_model,
        features=best_params['features'],
        train_smoothed=[hard_spots_train_smoothed, hard_spots_validation_smoothed]
    )

    return best_clf, best_features


def print_metrics(y_true, y_pred, label="Set"):
    print(f"\n=== {label} Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score (macro):  {f1_score(y_true, y_pred, average='macro'):.4f}")
