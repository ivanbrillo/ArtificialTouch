import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from plotter import plot_confusion_matrix
import numpy as np

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


# Plot comparison bar chart
def plot_model_comparison(results):
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(results)
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
    plt.title('Model Comparison')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()