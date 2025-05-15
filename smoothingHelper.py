import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import median_filter, gaussian_filter


def get_grid_bounds(df):
    x_min, x_max = int(df["posx"].min()), int(df["posx"].max())
    y_min, y_max = int(df["posy"].min()), int(df["posy"].max())
    grid_shape = (y_max - y_min + 1, x_max - x_min + 1)
    return x_min, y_min, grid_shape


def interpolate_subset_simple_labels(subset_df, feature_list):
    x_min, y_min, (H, W) = get_grid_bounds(subset_df)
    coords = [(yi, xi) for yi in range(H) for xi in range(W)]
    posx = [xi + x_min for _, xi in coords]
    posy = [yi + y_min for yi, _ in coords]
    full = pd.DataFrame({'posx': posx, 'posy': posy})

    # 2) interpolate each feature as you already do…
    for feature in feature_list:
        grid_z = np.full((H, W), np.nan, dtype=np.float32)
        for _, row in subset_df.iterrows():
            xi = int(row['posx']) - x_min
            yi = int(row['posy']) - y_min
            grid_z[yi, xi] = row[feature]
        yy, xx = np.indices((H, W))
        mask = ~np.isnan(grid_z)
        if np.any(mask):
            interp = NearestNDInterpolator(
                np.column_stack((yy[mask], xx[mask])),
                grid_z[mask]
            )
            grid_z = interp(yy, xx)
        full[feature] = grid_z.ravel()

    # 3) build label_grid with NaNs
    label_grid = np.full((H, W), np.nan, dtype=float)
    for _, row in subset_df.iterrows():
        xi = int(row['posx']) - x_min
        yi = int(row['posy']) - y_min
        label_grid[yi, xi] = row['label']  # could be 0 or >0

    for yi, xi in coords:
        if np.isnan(label_grid[yi, xi]):
            neighbors = []

            if xi > 0 and not np.isnan(label_grid[yi, xi - 1]):
                neighbors.append(label_grid[yi, xi - 1])  # left
            if xi < W - 1 and not np.isnan(label_grid[yi, xi + 1]):
                neighbors.append(label_grid[yi, xi + 1])  # right
            if yi > 0 and not np.isnan(label_grid[yi - 1, xi]):
                neighbors.append(label_grid[yi - 1, xi])  # up
            if yi < H - 1 and not np.isnan(label_grid[yi + 1, xi]):
                neighbors.append(label_grid[yi + 1, xi])  # down

            if neighbors:
                label_grid[yi, xi] = np.median(neighbors)

    # 5) attach back to DataFrame
    full['label'] = label_grid.ravel()
    return full


def apply_smoothing(grid_z, method='median'):
    if method == 'median':
        return median_filter(grid_z, size=3, mode='reflect')
    elif method == 'gaussian':
        return gaussian_filter(grid_z, sigma=2)
    else:
        raise ValueError(f"Smoothing method not implemented: {method}")


# Smoothing function for a single dataframe
def smooth_subset(subset_df, feature_list, smoothing_config):
    x_min, y_min, grid_shape = get_grid_bounds(subset_df)
    smoothed_subset = pd.DataFrame(index=subset_df.index)

    for feature in feature_list:
        # Initialize grid with NaNs
        grid_z = np.full(grid_shape, np.nan, dtype=np.float32)

        # Map each data point to the grid
        for _, row in subset_df.iterrows():
            x_idx = int(row["posx"]) - x_min
            y_idx = int(row["posy"]) - y_min
            grid_z[y_idx, x_idx] = row[feature]

        # Apply feature-specific smoothing
        methods = smoothing_config.get(feature,
                                       'median')  # Get the designed methods, if None select defaul 'median' method
        if isinstance(methods, list):
            for method in methods:
                grid_z = apply_smoothing(grid_z, method=method)
        else:
            grid_z = apply_smoothing(grid_z, method=methods)

        # Map smoothed grid back to DataFrame
        smoothed_subset[feature] = [
            grid_z[int(row["posy"]) - y_min, int(row["posx"]) - x_min]
            for _, row in subset_df.iterrows()
        ]

    # Add back metadata columns
    smoothed_subset[['label', 'posx', 'posy']] = subset_df[['label', 'posx', 'posy']]
    return smoothed_subset
