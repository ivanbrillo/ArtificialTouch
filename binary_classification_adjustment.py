import numpy as np

def adjust_test_features(X_train, y_train, X_test, y_pred):
    zero_train = X_train[(y_train == 0)]
    zero_test = X_test[(y_pred == 0)]

    # Compute the mean for each feature of the zero-th class
    feature_means_train = np.mean(zero_train, axis=0)
    feature_means_test = np.mean(zero_test, axis=0)

    mean_differences = feature_means_train - feature_means_test

    # Adjust the test set features
    adjusted_X_test = X_test + mean_differences

    return adjusted_X_test


