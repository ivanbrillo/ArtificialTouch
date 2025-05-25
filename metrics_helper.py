import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, f1_score


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
                y_true, y_proba, average='macro', multi_class='ovr'
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
