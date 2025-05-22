from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(labels, ax, title):
    """Plots a histogram of class distribution on the given axis."""
    class_counts = Counter(labels)
    labels_sorted = sorted(class_counts.keys())
    counts = [class_counts[l] for l in labels_sorted]
    ax.bar(labels_sorted, counts, color=sns.color_palette("turbo", len(labels_sorted)))
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")


def visualize_datasets(data_train, data_test, data_val, label_column='label'):
    """Prints class distribution and plots spatial and label distribution for datasets."""

    def print_class_distribution(dataset, name):
        dist = Counter(dataset[label_column])
        total = len(dataset)
        print(f"\nClass distribution in {name}:")
        for label, count in sorted(dist.items()):
            print(f"Label {label}: {count} samples ({count / total * 100:.2f}%)")

    # Print class distributions
    print_class_distribution(data_train, "Training set")
    print_class_distribution(data_test, "Test set")
    print_class_distribution(data_val, "Validation set")

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={'width_ratios': [3, 1]})
    sns.set(style="whitegrid")

    # Consistent color palette
    unique_labels = sorted(set(data_train[label_column]) |
                           set(data_test[label_column]) |
                           set(data_val[label_column]))
    palette_colors = sns.color_palette("turbo", n_colors=len(unique_labels))
    label_color_dict = {label: color for label, color in zip(unique_labels, palette_colors)}

    datasets = [
        ("Train", data_train, axs[0]),
        ("Test", data_test, axs[1]),
        ("Validation", data_val, axs[2])
    ]

    for title, data, (scatter_ax, hist_ax) in datasets:
        sns.scatterplot(
            data=data,
            x='posx',
            y='posy',
            hue=label_column,
            palette=label_color_dict,
            ax=scatter_ax
        )
        scatter_ax.set_title(f"{title} Set Distribution (posx vs posy)")
        scatter_ax.set_xlabel("Position X (posx)")
        scatter_ax.set_ylabel("Position Y (posy)")
        scatter_ax.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

        plot_class_distribution(data[label_column], hist_ax, f"Class Dist. ({title})")

    plt.tight_layout()
    plt.show()
