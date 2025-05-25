import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set(style="whitegrid")


def plot_features(df1, df2, feature_list):
    for feature in feature_list:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 3 columns, 1 row

        # Extract unique labels
        classes = sorted(df1["label"].unique())  # Assuming labels are the same in both datasets

        # Prepare boxplot data for each class
        data1_box = [df1[df1["label"] == lbl][feature].dropna() for lbl in classes]
        data2_box = [df2[df2["label"] == lbl][feature].dropna() for lbl in classes]

        # Boxplot for both datasets (stacked & colored)
        box1 = axes[0].boxplot(data1_box, positions=np.arange(len(classes)) - 0.2, widths=0.3, patch_artist=True,
                               boxprops=dict(facecolor="lightblue"), medianprops=dict(color="black"))
        box2 = axes[0].boxplot(data2_box, positions=np.arange(len(classes)) + 0.2, widths=0.3, patch_artist=True,
                               boxprops=dict(facecolor="lightcoral"), medianprops=dict(color="black"))

        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes)
        axes[0].set_title(f"Boxplot of {feature} by Label (Data1 vs Data2)")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel(feature)
        axes[0].legend([box1["boxes"][0], box2["boxes"][0]], ["Data1", "Data2"], loc="upper right")

        # Scatter plot limits
        vmin, vmax = np.percentile(df1[feature].dropna(), [0.2, 99.8])
        vmin2, vmax2 = np.percentile(df2[feature].dropna(), [0.2, 99.8])
        vmin = min(vmin, vmin2)
        vmax = max(vmax, vmax2)

        # Scatter plot for df1 (Data1)
        df1_label_0 = df1[df1["label"] == 0]
        df1_other_labels = df1[df1["label"] != 0]

        axes[1].scatter(df1_label_0["posx"], df1_label_0["posy"], c=np.clip(df1_label_0[feature], vmin, vmax),
                        cmap="viridis", vmin=vmin, vmax=vmax, marker='o', label="Label 0")
        axes[1].scatter(df1_other_labels["posx"], df1_other_labels["posy"],
                        c=np.clip(df1_other_labels[feature], vmin, vmax),
                        cmap="viridis", vmin=vmin, vmax=vmax, marker='x', label="Other Labels")
        axes[1].set_xlabel("Posx")
        axes[1].set_ylabel("Posy")
        axes[1].set_title(f"2D Plot of {feature} (Data1)")
        axes[1].legend()

        # Scatter plot for df2 (Data2)
        df2_label_0 = df2[df2["label"] == 0]
        df2_other_labels = df2[df2["label"] != 0]

        axes[2].scatter(df2_label_0["posx"], df2_label_0["posy"], c=np.clip(df2_label_0[feature], vmin, vmax),
                        cmap="viridis", vmin=vmin, vmax=vmax, marker='o', label="Label 0")
        axes[2].scatter(df2_other_labels["posx"], df2_other_labels["posy"],
                        c=np.clip(df2_other_labels[feature], vmin, vmax),
                        cmap="viridis", vmin=vmin, vmax=vmax, marker='x', label="Other Labels")
        axes[2].set_xlabel("Posx")
        axes[2].set_ylabel("Posy")
        axes[2].set_title(f"2D Plot of {feature} (Data2)")
        axes[2].legend()

        # Adjust layout
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(y_true, y_pred, title, ax, class_labels=[0, 1, 2, 3, 4, 5]):
    classes = class_labels

    # Create confusion matrix with specific labels
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Handle division by zero - create a copy to avoid modifying the original
    row_sums = cm.sum(axis=1)
    cm_norm = np.zeros_like(cm, dtype=float)

    # Only normalize rows that have samples
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum
        # rows with sum=0 remain as zeros

    # Plot the matrix
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')


# Plot the smoothing effect for comparison
def plot_smoothing_effect(validation_original, validation_smooth, train_original, train_smoothed, feature):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Plot original values
    sc1 = axes[0].scatter(train_original['posx'], train_original['posy'],
                          c=train_original[feature], cmap='viridis',
                          s=50, alpha=0.8)
    axes[0].set_title(f"Train Original {feature}")
    axes[0].set_xlabel("posx")
    axes[0].set_ylabel("posy")

    # Plot smoothed values
    sc2 = axes[1].scatter(train_smoothed['posx'], train_smoothed['posy'],
                          c=train_smoothed[feature], cmap='viridis',
                          s=50, alpha=0.8)
    axes[1].set_title(f"Train Max Smoothed {feature}")
    axes[1].set_xlabel("posx")
    axes[1].set_ylabel("posy")

    # Validation set
    sc3 = axes[2].scatter(validation_original['posx'], validation_original['posy'],
                          c=validation_original[feature], cmap='viridis',
                          s=50, alpha=0.8)
    axes[2].set_title(f"Validation Original {feature}")
    axes[2].set_xlabel("posx")
    axes[2].set_ylabel("posy")

    sc4 = axes[3].scatter(validation_smooth['posx'], validation_smooth['posy'],
                          c=validation_smooth[feature], cmap='viridis',
                          s=50, alpha=0.8)
    axes[3].set_title(f"Validation Max Smoothed {feature}")
    axes[3].set_xlabel("posx")
    axes[3].set_ylabel("posy")

    plt.tight_layout()
    plt.show()


def plot_prediction_map(X_data, y_true, y_pred, title="Prediction Map"):
    # Create masks for correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    incorrect_mask = ~correct_mask

    # Generate a distinct color for each class
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    num_classes = len(unique_classes)
    cmap = cm.get_cmap('tab10', num_classes)
    class_colors = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(11, 11))

    # Plot correctly predicted points
    for cls in unique_classes:
        mask = (y_true == cls) & correct_mask
        ax.scatter(X_data.loc[mask, 'posx'],
                   X_data.loc[mask, 'posy'],
                   color=class_colors[cls],
                   label=f'Class {cls} - Correct',
                   marker='o',
                   s=60,
                   alpha=0.7)

    # Plot incorrectly predicted points
    for cls in unique_classes:
        mask = (y_true == cls) & incorrect_mask
        ax.scatter(X_data.loc[mask, 'posx'],
                   X_data.loc[mask, 'posy'],
                   color=class_colors[cls],
                   label=f'Class {cls} - Incorrect',
                   marker='x',
                   s=80,
                   alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("posx")
    ax.set_ylabel("posy")
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
        ncol=6,
        frameon=False
    )
    plt.tight_layout()
