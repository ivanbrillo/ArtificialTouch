import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.feature_selection import mutual_info_classif


def calculate_mutual_info(X_data, y_data, feature_list, random_state=42):
    """Calculate mutual information scores for features."""
    return mutual_info_classif(X_data[feature_list], y_data, random_state=random_state)


def calculate_wasserstein_distances(train_data, validation_data, feature_list):
    """Calculate Wasserstein distance between train and validation for each feature."""
    distances = []

    for feature in feature_list:
        distance = wasserstein_distance(train_data[feature], validation_data[feature])
        distances.append({'feature': feature, 'wasserstein_distance': distance})

    return pd.DataFrame(distances)


def calculate_distribution_metrics(train_data, validation_data, feature_list):
    """Calculate distribution metrics between train and validation sets."""
    metrics = []

    for feature in feature_list:
        # Wasserstein distance (Earth Mover's Distance)
        w_distance = wasserstein_distance(train_data[feature], validation_data[feature])

        metrics.append({
            'feature': feature,
            'wasserstein_distance': w_distance,
        })

    return pd.DataFrame(metrics)


def calculate_class_distribution_metrics(X_train, y_train, X_validation, y_validation, feature_list, selected_classes):
    """Calculate distribution metrics for each selected class."""
    class_metrics = {}

    for cls in selected_classes:
        # Filter data for this class
        X_train_cls = X_train[y_train == cls]
        X_val_cls = X_validation[y_validation == cls]

        # Calculate metrics for this class
        metrics_df = calculate_distribution_metrics(X_train_cls, X_val_cls, feature_list)
        class_metrics[cls] = metrics_df

    return class_metrics


def filter_by_classes(X, y, classes):
    mask = y.isin(classes)
    return X[mask], y[mask]


def compute_mi_dataframe(X_train, y_train, X_val, y_val, features):
    train_scores = calculate_mutual_info(X_train, y_train, features)
    val_scores = calculate_mutual_info(X_val, y_val, features)
    return pd.DataFrame({
        'feature': features,
        'train': train_scores,
        'validation': val_scores
    })


def compute_wasserstein_metrics(X_train, X_val, y_train, y_val, features, classes):
    overall = calculate_distribution_metrics(X_train, X_val, features)
    per_class = calculate_class_distribution_metrics(
        X_train, y_train, X_val, y_val, features, classes
    )
    return overall, per_class
