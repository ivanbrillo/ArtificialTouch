import pandas as pd
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from featureSelectionHelper import filter_by_classes, compute_mi_dataframe, compute_wasserstein_metrics
from featureSelectionPlotter import plot_mi_wdist, plot_wdist_class


def select_features(mi_df, metrics_df, class_metrics, selected_classes,
                    mi_threshold=0.5, wasserstein_threshold=0.12,
                    class_wasserstein_threshold=None):
    """
    Select features based on multiple criteria:
    1. High mutual information in both train and validation
    2. Low overall distribution distances between train and validation
    3. Low per-class distribution distances
    """
    # Step 1: Filter by mutual information
    mi_selected = mi_df[(mi_df['train'] > mi_threshold) &
                        (mi_df['validation'] > mi_threshold)]

    # Step 2: Filter by overall distribution metrics
    dist_selected = metrics_df[metrics_df['wasserstein_distance'] < wasserstein_threshold]

    # Combine the filters
    combined = mi_selected[['feature']].merge(dist_selected[['feature']], on='feature')
    combined_features = combined['feature'].tolist()

    # Step 3: Filter by per-class distribution metrics
    if class_wasserstein_threshold is not None:
        final_features = []
        for feature in combined_features:
            use_feature = True
            for cls in selected_classes:
                cls_df = class_metrics[cls]
                w_dist = cls_df.loc[cls_df['feature'] == feature, 'wasserstein_distance'].values[0]
                if w_dist > class_wasserstein_threshold:
                    use_feature = False
                    break
            if use_feature:
                final_features.append(feature)
    else:
        final_features = combined_features

    return final_features


def find_best_features(
        X_train_scaled,
        y_train,
        X_validation_scaled,
        y_validation,
        feature_list,
        selected_classes,
        mi_threshold=0.5,
        wasserstein_threshold=0.1,
        class_wasserstein_threshold=None,
        plot=False
):
    # 1. Filter
    X_train_filt, y_train_filt = filter_by_classes(
        X_train_scaled, y_train, selected_classes
    )
    X_val_filt, y_val_filt = filter_by_classes(
        X_validation_scaled, y_validation, selected_classes
    )

    # 2. Mutual Information DF
    mi_df = compute_mi_dataframe(
        X_train_filt, y_train_filt, X_val_filt, y_val_filt, feature_list
    )

    # 3. Distribution metrics with Wasserstein distance
    metrics_df, class_metrics = compute_wasserstein_metrics(
        X_train_filt, X_val_filt, y_train_filt, y_val_filt,
        feature_list, selected_classes
    )

    # 4. Feature selection
    selected = select_features(
        mi_df,
        metrics_df,
        class_metrics,
        selected_classes,
        mi_threshold,
        wasserstein_threshold,
        class_wasserstein_threshold
    )

    # 5. Optional plotting
    if plot:
        plot_mi_wdist(mi_df, metrics_df)
        plot_wdist_class(
            X_train_filt,
            y_train_filt,
            X_val_filt,
            y_val_filt,
            selected,
            selected_classes
        )

    return selected


def find_best_features_sets(
        X_train,
        y_train,
        X_validation,
        y_validation,
        features_to_smooth,
        mi_th,
        wasserstein_th,
        wasserstein_th_per_class,
        classes_to_check
):
    """
    For each (w_th,w_th_c) in the grid, smooth the df, then compute silhouette_score on the smoothed feature matrix in order to select
    just one feature set in the ones with same number of features.
    """
    results = []

    for w_th in tqdm(wasserstein_th, desc="bandwidth"):
        for w_th_c in wasserstein_th_per_class:
            # 1) smooth
            selected_features = find_best_features(
                X_train_scaled=X_train,
                y_train=y_train,
                X_validation_scaled=X_validation,
                y_validation=y_validation,
                feature_list=features_to_smooth,
                selected_classes=classes_to_check,
                mi_threshold=mi_th,
                wasserstein_threshold=w_th,
                class_wasserstein_threshold=w_th_c
            )

            if len(selected_features) == 0:
                continue

            X_pd = pd.concat(
                [X_train[selected_features], X_validation[selected_features]],
                ignore_index=True
            )

            true_labels = pd.concat(
                [y_train, y_validation],
                ignore_index=True
            )

            score = silhouette_score(X_pd.values, true_labels.values)

            results.append({
                'w_th': w_th,
                'w_th_c': w_th_c,
                'n_features': len(selected_features),
                'features': selected_features,
                'silhouette': score
            })

    results_df = pd.DataFrame(results)

    # keeps the best silhouette score for equal number of features
    results_df = results_df.loc[results_df.groupby('n_features')['silhouette'].idxmax()].reset_index(drop=True)
    results_df = results_df[results_df["silhouette"] > 0].reset_index(drop=True)

    return results_df
