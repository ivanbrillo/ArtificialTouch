from statistics import correlation
import pandas as pd
import glob
import numpy as np
import json
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.interpolate import griddata
from scipy.signal import coherence
from skimage.feature import local_binary_pattern
from sklearn.neighbors import LocalOutlierFactor
from scipy.optimize import curve_fit
from pywt import wavedec
from scipy.signal import stft
from PyEMD import EMD
from neurokit2 import complexity
from scipy import ndimage
from skimage import measure, morphology
import antropy
import nolds
from scipy.ndimage import sobel, laplace
from skimage.feature import hog

materials_label = {
    "Dragon Skin shore 20A": 1,
    "Dragon Skin shore 30A": 2,
    "SORTA-Clear shore 40A": 3,
    "PDMS shore 44": 4,
    "Econ shore 80A": 5
}


def extract_features(data):
    # Gaussian Smoothing
    data['Fz_s'] = gaussian_filter1d(data['forceZ'], sigma=2)
    data['posz_s'] = gaussian_filter1d(data['posz'], sigma=2)

    pz = (data['posz_s'][data['isArrived_Festo'] == 1] / 1000).to_numpy()  # From mm to m
    fz = (data['Fz_s'][data['isArrived_Festo'] == 1]).to_numpy()
    time = data['CPXEts'][data['isArrived_Festo'] == 1].to_numpy()

    fz_touch = (data['Fz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    pz_touch = (data['posz_s'][data['isTouching_SMAC'] == 1]).to_numpy()
    time_touch = data['CPXEts'][data['isTouching_SMAC'] == 1].to_numpy()

    if len(time) == 0:
        return None

    time_i = np.arange(time[0], time[-1], 0.001)
    time_t = np.arange(time_touch[0], time_touch[-1], 0.001)

    f = interp1d(time, fz, kind='cubic')
    f2 = interp1d(time_touch, fz_touch, kind='cubic')
    fi = f(time_i)
    ft = f2(time_t)

    g = interp1d(time, pz, kind='cubic')
    g2 = interp1d(time_touch, pz_touch, kind='cubic')
    pi = g(time_i)
    pt = g2(time_t)

    # Calculate key indices once - reuse them throughout
    force_max = np.max(fi)
    force_max_idx = np.argmax(fi)
    steady_state_start_idx = min(force_max_idx + int(len(fi) * 0.05), len(fi) - 101)
    steady_state_force = np.mean(
        fi[steady_state_start_idx:steady_state_start_idx + 100]) if steady_state_start_idx < len(fi) - 100 else np.mean(
        fi[-100:])

    # Calculate differences and derivatives once (reuse for multiple features)
    dt = np.mean(np.diff(time_i))
    velocity = np.gradient(pi, time_i)

    # Calculate force thresholds once
    low_threshold = 0.1 * force_max
    mid_threshold = 0.5 * force_max
    high_threshold = 0.75 * force_max

    # Force Decay Rate (τ)
    decay_time = time_t - time_t[0]
    decay_force = np.log(np.maximum(ft, 1e-10))  # Avoid log(0)
    tau_result = linregress(decay_time, decay_force)
    tau = -1 / tau_result.slope if tau_result.slope != 0 else None

    # Steady-State Force
    F_ss = steady_state_force
    P_ss = np.mean(pt[-30:]) if len(pt) >= 30 else None

    # Calculate stiffness using pre-computed thresholds
    stiffness = None
    upstroke = None
    downstroke = None

    # Find indices efficiently
    low_idx_up = np.where(fi >= low_threshold)[0]
    mid_idx_up = np.where(fi >= mid_threshold)[0]
    high_idx_up = np.where(fi >= high_threshold)[0]

    post_peak_indices = np.where((fi <= high_threshold) & (np.arange(len(fi)) > force_max_idx))[0]
    down_end_indices = np.where((fi <= low_threshold) & (np.arange(len(fi)) > force_max_idx))[0]

    if len(low_idx_up) > 0:
        up_start_idx = low_idx_up[0]

        # Calculate stiffness
        if pi[force_max_idx] != pi[up_start_idx]:
            stiffness = (force_max - fi[up_start_idx]) / (pi[force_max_idx] - pi[up_start_idx])

        # Calculate upstroke
        if len(mid_idx_up) > 0:
            mid_up_idx = mid_idx_up[0]
            if pi[mid_up_idx] != pi[up_start_idx]:
                upstroke = (fi[mid_up_idx] - fi[up_start_idx]) / (pi[mid_up_idx] - pi[up_start_idx])

        # Calculate downstrokes
        if len(post_peak_indices) > 0:
            down1_idx = post_peak_indices[0]
            if len(down_end_indices) > 0:
                down_end_idx = down_end_indices[0]
                if pi[down1_idx] != pi[down_end_idx]:
                    downstroke = (fi[down1_idx] - fi[down_end_idx]) / (pi[down1_idx] - pi[down_end_idx])


    # Entropy
    # Use fewer bins for faster computation
    hist, _ = np.histogram(ft, bins=20, density=True)
    hist_positive = hist[hist > 0]
    entropy = -np.sum(hist_positive * np.log2(hist_positive)) if len(hist_positive) > 0 else None

    # Additional features
    time_to_max = time_i[force_max_idx] - time_i[0]
    force_overshoot = (force_max - F_ss) / F_ss if F_ss > 0 else 0


    # Force relaxation
    if len(ft) > 100:
        force_relaxation = (force_max - np.mean(ft[-50:])) / force_max if force_max > 0 else 0
    else:
        force_relaxation = 0

    # Force/position RMS
    force_rms = np.sqrt(np.mean(np.square(fi)))
    position_rms = np.sqrt(np.mean(np.square(pi)))

    # Damping coefficient (optimized to avoid singular matrix errors)
    try:
        A = np.vstack([pi, velocity]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, fi, rcond=1e-10)
        elastic_coefficient, damping_coefficient = coeffs
    except:
        elastic_coefficient, damping_coefficient = None, None

    # Contact area estimation
    contact_area = None
    if stiffness and stiffness > 0:
        R = 0.005  # Estimated radius in meters
        nu = 0.5  # Typical Poisson's ratio
        contact_area = np.pi * ((3 * (1 - nu ** 2) * force_max * R / (4 * stiffness)) ** (2 / 3))

    # Force peak-to-peak
    force_ptp = np.ptp(fi)

    # Position peak-to-peak
    position_ptp = np.ptp(pi)

    # Load duration calculation
    load_duration = time_i[force_max_idx] - time_i[0]

    # Hjorth parameters
    activity = np.var(fi)
    mobility = np.sqrt(np.var(np.diff(fi)) / activity) if activity > 0 else None
    complexity_value = np.sqrt(np.var(np.diff(np.diff(fi))) / np.var(np.diff(fi))) / mobility if mobility and np.var(
        np.diff(fi)) > 0 else None

    # Teager-Kaiser Energy Operator (simplified)
    tkeo_mean = None
    try:
        tkeo = fi[1:-1] ** 2 - fi[:-2] * fi[2:]
        tkeo_mean = np.mean(tkeo)
    except:
        pass

    # Correlation between force and position
    correlation_fp = np.corrcoef(fi, pi)[0, 1] if len(fi) > 1 and len(pi) > 1 else None

    # Calculate peak ratio
    peak_ratio = force_max / np.max(pi) if np.max(pi) > 0 else None

    # Ensure signal has sufficient variance for meaningful analysis
    if np.std(fi) > 1e-6:
        try:
            # Sample_entropy
            sample_entropy = antropy.sample_entropy(fi)
        except Exception:
            sample_entropy = np.nan
    else:
        sample_entropy = np.nan

    # Force Oscillation
    # Calculate standard deviation in force during steady state as a measure of oscillations
    steady_state_region = fi[steady_state_start_idx:steady_state_start_idx + 150]
    force_oscillation = np.std(steady_state_region)


    # Return all features in the original tuple format
    return (stiffness, tau, F_ss, entropy, upstroke, downstroke, fi, pi, time_i, P_ss,
            force_max, time_to_max, force_overshoot, force_relaxation,
            force_rms, position_rms, damping_coefficient, elastic_coefficient, contact_area,
            force_ptp, position_ptp, load_duration,
            activity, mobility, complexity_value,
            tkeo_mean, correlation_fp, peak_ratio,sample_entropy,force_oscillation)

def extract_curve_features(position, force):
    features = {}

    # --- 1. Peak detection ---
    peak_idx = np.argmax(force)
    features['peak_position'] = position[peak_idx]

    # --- 2. Hysteresis area ---
    # Ensure the curve is closed (append first point if necessary)
    hysteresis_area = np.nan
    if not np.array_equal(position[0], position[-1]) or not np.array_equal(force[0], force[-1]):
        position_closed = np.append(position, position[0])
        force_closed = np.append(force, force[0])
    else:
        position_closed = position
        force_closed = force
        hysteresis_area = 0.5 * abs(np.sum(
        position_closed[:-1] * force_closed[1:] - position_closed[1:] * force_closed[:-1]
    ))
        features['hysteresis_area'] = hysteresis_area

    # --- 3. Loading energy (area under loading curve) ---
    loading_energy = np.trapezoid(force[:peak_idx + 1], position[:peak_idx + 1])
    features['loading_energy'] = loading_energy

    # --- 4. Loading nonlinearity ---
    if peak_idx > 10:
        try:
            quad_fit = np.polyfit(position[:peak_idx + 1], force[:peak_idx + 1], 2)
            features['loading_nonlinearity'] = abs(quad_fit[0])
        except Exception:
            features['loading_nonlinearity'] = np.nan
    else:
        features['loading_nonlinearity'] = np.nan

    # --- 5. Loading-to-unloading area ratio ---
    if peak_idx > 10 and peak_idx < len(position) - 10:
        loading_area = np.trapezoid(force[:peak_idx + 1], position[:peak_idx + 1])
        unloading_area = np.trapezoid(force[peak_idx:], position[peak_idx:])
        features['loading_unloading_area_ratio'] = loading_area / abs(unloading_area) if unloading_area != 0 else np.nan
    else:
        features['loading_unloading_area_ratio'] = np.nan

    # --- 6. Force ratio (75% displacement / 25% displacement) ---
    if peak_idx > 5:
        peak_pos = position[peak_idx]
        idx_25 = np.argmin(np.abs(position[:peak_idx + 1] - 0.25 * peak_pos))
        idx_75 = np.argmin(np.abs(position[:peak_idx + 1] - 0.75 * peak_pos))
        f25 = force[idx_25]
        f75 = force[idx_75]
        features['force_ratio_75_25'] = f75 / f25 if f25 != 0 else np.nan
    else:
        features['force_ratio_75_25'] = np.nan

    # --- 7. Loading skewness and kurtosis ---
    if peak_idx > 5:
        features['loading_skewness'] = skew(force[:peak_idx + 1])
        features['loading_kurtosis'] = kurtosis(force[:peak_idx + 1])
    else:
        features['loading_skewness'] = np.nan
        features['loading_kurtosis'] = np.nan

    # --- 8. Polynomial curve fits ---
    if peak_idx > 15:
        x_load = position[:peak_idx + 1]
        y_load = force[:peak_idx + 1]
        x_norm = (x_load - np.min(x_load)) / (np.ptp(x_load) if np.ptp(x_load) != 0 else 1)
        y_norm = y_load / np.max(y_load) if np.max(y_load) != 0 else y_load

        for degree in [3, 4, 5]:
            try:
                poly_fit = np.polyfit(x_norm, y_norm, degree)
                for i, coef in enumerate(poly_fit):
                    features[f'poly{degree}_coef{i}'] = coef
            except Exception:
                for i in range(degree + 1):
                    features[f'poly{degree}_coef{i}'] = np.nan

        try:
            cubic_fit = np.polyfit(x_load, y_load, 3)
            features['cubic_coefficient'] = cubic_fit[0]
        except Exception:
            features['cubic_coefficient'] = np.nan
        try:
            quartic_fit = np.polyfit(x_load, y_load, 4)
            features['quartic_coefficient'] = quartic_fit[0]
        except Exception:
            features['quartic_coefficient'] = np.nan
    else:
        for degree in [3, 4, 5]:
            for i in range(degree + 1):
                features[f'poly{degree}_coef{i}'] = np.nan
        features['cubic_coefficient'] = np.nan
        features['quartic_coefficient'] = np.nan

    # --- 9. Segmentation features (only segments 2 and 3) ---
    if peak_idx > 15:
        num_segments = 4
        segment_size = (peak_idx + 1) // num_segments
        for seg in [2, 3]:
            start = seg * segment_size
            end = (seg + 1) * segment_size if seg < num_segments - 1 else peak_idx + 1
            if end - start > 5:
                seg_pos = position[start:end]
                seg_force = force[start:end]
                slope = (seg_force[-1] - seg_force[0]) / (seg_pos[-1] - seg_pos[0]) if (seg_pos[-1] - seg_pos[
                    0]) != 0 else np.nan
                std = np.std(seg_force)
                seg_skew = skew(seg_force) if len(seg_force) > 3 else np.nan
            else:
                slope = np.nan
                std = np.nan
                seg_skew = np.nan
            features[f'segment{seg}_slope'] = slope
            features[f'segment{seg}_force_std'] = std
            features[f'segment{seg}_skew'] = seg_skew
    else:
        for seg in [2, 3]:
            features[f'segment{seg}_slope'] = np.nan
            features[f'segment{seg}_force_std'] = np.nan
            features[f'segment{seg}_skew'] = np.nan

    return features

def compute_hysteresis_features_for_df(df, force_column='Fi', pos_column='Pi', threshold=0.1):
    """
    Given a DataFrame (with force and position data stored as arrays/lists in each row),
    compute the hysteresis features using extract_curve_features() and add them as new columns.
    """
    # List of hysteresis features (update if additional features are added in extract_curve_features)
    hysteresis_features = [
        'peak_position', 'hysteresis_area', 'loading_energy', 'loading_nonlinearity',
        'loading_unloading_area_ratio', 'force_ratio_75_25', 'loading_skewness',
        'loading_kurtosis', 'poly3_coef0', 'poly3_coef1', 'poly3_coef2', 'poly3_coef3',
        'poly4_coef0', 'poly4_coef1', 'poly4_coef2', 'poly4_coef3', 'poly4_coef4',
        'poly5_coef0', 'poly5_coef1', 'poly5_coef2', 'poly5_coef3', 'poly5_coef4',
        'segment2_slope', 'segment3_slope', 'segment2_force_std', 'segment3_force_std',
        'segment2_skew', 'segment3_skew', 'cubic_coefficient', 'quartic_coefficient']

    # Initialize the hysteresis feature columns if they don't exist
    for feat in hysteresis_features:
        if feat not in df.columns:
            df[feat] = np.nan

    # Loop through each row and compute the features
    for idx, row in df.iterrows():
        force = np.array(row[force_column])
        position = np.array(row[pos_column])
        # Align position so that the first force value above the threshold is at zero
        force_above_threshold = force > threshold
        if np.any(force_above_threshold):

            start_idx = np.where(force_above_threshold)[0][0]
            aligned_position = position - position[start_idx]
            feats = extract_curve_features(aligned_position, force)
            for key, value in feats.items():
                df.at[idx, key] = value
    return df

def compute_spatial_and_surface_features(df):
    """
    Combined function to compute spatial, surface and LBP features across the 2D grid with interpolation
    for missing points but without adding features for those missing points.
    """

    # Calculate grid dimensions from data
    minx = int(df['posx'].min())
    miny = int(df['posy'].min())
    maxx = int(df['posx'].max())
    maxy = int(df['posy'].max())

    # Adjust grid size based on data boundaries
    width = maxx - minx + 1
    height = maxy - miny + 1

    # Create a list of features to analyze
    features_list = ['Stiffness', 'Upstroke', 'Downstroke','Tau','time_to_max']
    # Get the set of points that actually exist in the data
    existing_points = set(zip(df['posx'].astype(int), df['posy'].astype(int)))

    # Create grids for spatial analysis with interpolation for calculations
    grids = {}
    for feature in features_list:
        # Create initial grid with NaN values
        grid = np.full((height, width), np.nan)

        # Fill with available data
        for idx, row in df.iterrows():
            x, y = int(row['posx']) - minx, int(row['posy']) - miny
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = row[feature]

        # Interpolate missing values for calculation purposes
        filled_grid = interpolate_missing_values(grid)
        grids[feature] = filled_grid

    # Create position grid for surface features
    position_grid = np.full((height, width), np.nan)
    for _, row in df.iterrows():
        x, y = int(row['posx']) - minx, int(row['posy']) - miny
        if 0 <= x < width and 0 <= y < height:
            position_grid[y, x] = row['P_ss']

    # Interpolate missing values in position grid
    filled_position_grid = interpolate_missing_values(position_grid)

    # Calculate global statistics (for deviation features)
    global_mean_stiffness = np.nanmean(grids['Stiffness'])
    global_mean_position = np.nanmean(filled_position_grid)
    global_std_stiffness = np.nanstd(grids['Stiffness'])
    global_std_position = np.nanstd(filled_position_grid)
    global_median_stiffness = np.nanmedian(grids['Stiffness'])
    global_median_position = np.nanmedian(filled_position_grid)


    # Only process for EXISTING points
    for idx, row in df.iterrows():
        orig_x, orig_y = int(row['posx']), int(row['posy'])
        point = (orig_x, orig_y)

        # Check if this point actually exists in the original data
        if (orig_x, orig_y) in existing_points:
            x, y = orig_x - minx, orig_y - miny

            # Only process if the point is within our grid
            if 0 <= x < width and 0 <= y < height:
                # Spatial features
                # Define 3x3 patch
                x_min, x_max = max(0, x - 1), min(width - 1, x + 1) + 1
                y_min, y_max = max(0, y - 1), min(height - 1, y + 1) + 1

                # Extract patches
                for feature in features_list:
                    patch = grids[feature][y_min:y_max, x_min:x_max].flatten()
                    patch = patch[~np.isnan(patch)]
                    if len(patch) > 0:
                        # Local mean
                        df.at[idx, f'local_mean_{feature}'] = np.mean(patch)

                        # Local std
                        df.at[idx, f'local_std_{feature}'] = np.std(patch)

                        # Local skewness
                        df.at[idx, f'local_skewness_{feature}'] = skew(patch)

                        # Local kurtosis
                        df.at[idx, f'local_kurtosis_{feature}'] = kurtosis(patch)

                        # Local range (max - min)
                        df.at[idx, f'local_range_{feature}'] = np.ptp(patch)


                # Deviation from global mean
                df.at[idx, 'stiffness_deviation_from_global'] = grids['Stiffness'][y, x] - global_mean_stiffness
                df.at[idx, 'position_deviation_from_global'] = filled_position_grid[y, x] - global_mean_position
                # Deviation from global median
                df.at[idx, 'stiffness_deviation_from_global_median'] = grids['Stiffness'][y, x] - global_median_stiffness
                df.at[idx, 'position_deviation_from_global_median'] = filled_position_grid[y, x] - global_median_position
                # Z-score relative to global distribution
                df.at[idx, 'stiffness_z_score'] = (grids['Stiffness'][y, x] - global_mean_stiffness) / global_std_stiffness
                df.at[idx, 'position_z_score'] = (filled_position_grid[y, x] - global_mean_position) / global_std_position

                # Local Contrast Normalization
                # Multi-scale contrast features with different window sizes
                for window_size in [3, 5, 7, 9]:
                    half_win = window_size // 2

                    # Define window boundaries with bounds checking
                    y_min_win = max(0, y - half_win)
                    y_max_win = min(height - 1, y + half_win) + 1
                    x_min_win = max(0, x - half_win)
                    x_max_win = min(width - 1, x + half_win) + 1

                    # Process multiple features
                    for feature in features_list:
                        patch = grids[feature][y_min_win:y_max_win, x_min_win:x_max_win]
                        valid_patch = patch[~np.isnan(patch)]

                        if len(valid_patch) > 0:
                            # Local contrast (max-min range normalized)
                            local_range = np.max(valid_patch) - np.min(valid_patch)
                            if local_range > 0:
                                range_norm_value = (grids[feature][y, x] - np.min(valid_patch)) / local_range
                                df.at[idx, f'{feature}_range_norm_w{window_size}'] = range_norm_value
                            else:
                                df.at[idx, f'{feature}_range_norm_w{window_size}'] = 0

                # Center-surround differences at multiple scales
                for center_size in [1, 2]:
                    for surround_size in [3, 5]:
                        if surround_size <= center_size:
                            continue

                        # Define center region
                        y_min_center = max(0, y - center_size)
                        y_max_center = min(height - 1, y + center_size) + 1
                        x_min_center = max(0, x - center_size)
                        x_max_center = min(width - 1, x + center_size) + 1

                        # Define surround region
                        y_min_surr = max(0, y - surround_size)
                        y_max_surr = min(height - 1, y + surround_size) + 1
                        x_min_surr = max(0, x - surround_size)
                        x_max_surr = min(width - 1, x + surround_size) + 1

                        # Process for stiffness feature
                        center_patch = grids['Stiffness'][y_min_center:y_max_center, x_min_center:x_max_center]
                        valid_center = center_patch[~np.isnan(center_patch)]

                        # Create mask for surround (excluding center)
                        surround_patch = grids['Stiffness'][y_min_surr:y_max_surr, x_min_surr:x_max_surr]
                        center_mask = np.zeros_like(surround_patch, dtype=bool)
                        center_mask[
                        y_min_center - y_min_surr:y_max_center - y_min_surr,
                        x_min_center - x_min_surr:x_max_center - x_min_surr
                        ] = True

                        surround_values = surround_patch[~center_mask]
                        valid_surround = surround_values[~np.isnan(surround_values)]

                        if len(valid_center) > 0 and len(valid_surround) > 0:
                            center_mean = np.mean(valid_center)
                            surround_mean = np.mean(valid_surround)

                            # Center-surround difference
                            cs_diff = center_mean - surround_mean
                            df.at[idx, f'center_surround_diff_c{center_size}_s{surround_size}'] = cs_diff

                            # Center-surround ratio
                            if surround_mean != 0:
                                cs_ratio = center_mean / surround_mean
                                df.at[idx, f'center_surround_ratio_c{center_size}_s{surround_size}'] = cs_ratio
                            else:
                                df.at[idx, f'center_surround_ratio_c{center_size}_s{surround_size}'] = np.nan

                            # Center-surround contrast
                            cs_contrast = abs(center_mean - surround_mean) / (center_mean + surround_mean + 1e-10)
                            df.at[idx, f'center_surround_contrast_c{center_size}_s{surround_size}'] = cs_contrast

                # LOCAL SURFACE METRICS - multiple window sizes
                window_sizes = [3, 5, 7]  # 3x3, 5x5, 7x7 windows

                for window_size in window_sizes:
                    half_window = window_size // 2
                    # Define neighborhood for surface metrics
                    y_min_surf = max(0, y - half_window)
                    y_max_surf = min(height - 1, y + half_window) + 1
                    x_min_surf = max(0, x - half_window)
                    x_max_surf = min(width - 1, x + half_window) + 1

                    # Extract neighborhood from filled position grid
                    local_positions = filled_position_grid[y_min_surf:y_max_surf, x_min_surf:x_max_surf].flatten()
                    local_positions = local_positions[~np.isnan(local_positions)]
                    # Extract neighborhood data for multiple features
                    local_stiffness = grids['Stiffness'][y_min_surf:y_max_surf, x_min_surf:x_max_surf].flatten()
                    # Remove NaN values
                    valid_indices = ~(np.isnan(local_stiffness) | np.isnan(local_positions))
                    local_stiffness = local_stiffness[valid_indices]

                    if len(local_positions) > 0:
                        # Surface roughness
                        local_mean = np.mean(local_positions)
                        df.at[idx, f'local_Ra_w{window_size}'] = np.mean(np.abs(local_positions - local_mean))
                        df.at[idx, f'local_Rq_w{window_size}'] = np.sqrt(np.mean(np.square(local_positions - local_mean)))

                        # Surface statistical properties
                        if len(local_positions) > 2:  # Need at least 3 points for skew and kurtosis
                            df.at[idx, f'local_skewness_w{window_size}'] = skew(local_positions)
                            df.at[idx, f'local_kurtosis_w{window_size}'] = kurtosis(local_positions)

                        # Calculate local max peak height and valley depth
                        if len(local_positions) > 1:
                            df.at[idx, f'local_peak_height_w{window_size}'] = np.max(local_positions) - local_mean
                            df.at[idx, f'local_valley_depth_w{window_size}'] = local_mean - np.min(local_positions)

    return df

def interpolate_missing_values(grid):
    """
    Interpolate missing values in a grid using nearest neighbor interpolation.
    """
    # Check if the grid has any non-NaN values
    if np.all(np.isnan(grid)):
        return grid.copy()

    # Get coordinates of non-NaN values
    y_indices, x_indices = np.where(~np.isnan(grid))
    valid_points = np.column_stack((x_indices, y_indices))
    valid_values = grid[y_indices, x_indices]

    # Create grid coordinates for interpolation
    y_grid, x_grid = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    all_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Interpolate
    interpolated_values = griddata(valid_points, valid_values, all_points, method='cubic')

    # Use nearest neighbor for remaining NaN values
    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        nearest_values = griddata(valid_points, valid_values, all_points[nan_mask], method='nearest')
        interpolated_values[nan_mask] = nearest_values

    # Reshape back to grid
    filled_grid = interpolated_values.reshape(grid.shape)

    return filled_grid

def find_inclusions(json_data, symmetric=False, symm_v=False, angle=0, offset=(49, 49)) -> tuple:
    circles = json_data["Inclusions"]

    # Define small clockwise rotation angle (-0.46 degrees)
    theta = np.radians(-0.46 + angle)  # Negative for clockwise rotation

    # Rotation matrix
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    # Rotation center
    center = np.array([100, 100])

    # Extract original centers
    c = np.array([(circle["Position"][0] + offset[0], circle["Position"][1] + offset[1]) for circle in circles])

    # Shift, rotate, and shift back **without rounding**
    c_shifted = c - center

    if symmetric:
        c_shifted = np.array([(-y, -x) for x, y in c_shifted])

    if symm_v:
        c_shifted = np.array([(-x, y) for x, y in c_shifted])


    c_rotated = np.dot(c_shifted, R)
    c_final = c_rotated + center + np.array([-1, -1])  # Keep as float

    # Convert back to list of tuples
    c_final_list = [tuple(point) for point in c_final]

    r = [circle["Diameter"] / 2 for circle in circles]
    materials = [circle["Material"] for circle in circles]

    return c_final_list, r, materials

def get_label(posx, posy, centers, radii, materials) -> int:
    for i in range(len(centers)):
        center_x, center_y = centers[i]
        radius = radii[i]
        material = materials[i]
        # Check if point is within the circle
        if ((posx - center_x) ** 2 + (posy - center_y) ** 2) <= (radius) ** 2:
            return materials_label[material]
            # return (i // 5) + 1  # Assign label based on group of 5 circles
    return 0  # Default label

def organize_df(df_input: DataFrame, centers, radii, materials) -> DataFrame | None:
    df_input['forceZ'] = np.where(abs(df_input["Fz"]) < 27648, df_input["Fz"] / 27648, df_input["Fz"] / 32767)
    offset = np.mean(df_input['forceZ'][df_input['isArrived_Festo'] == 1])
    # Force normalization
    df_input['forceZ'] = (df_input['forceZ'] - offset)  / np.mean(df_input['forceZ'][df_input['isTouching_SMAC'] == 1][ -30:])  # / np.std(df_input['forceZ'][df_input['isArrived_Festo'] == 1].tolist())   #  # /

    df_input['posz'] = df_input['posz'] + df_input['posz2'] / 200 + df_input['posz_d'] / 1000
    offset_p = np.mean(df_input['posz'][df_input['isArrived_Festo'] == 1])
    # Position normalization
    df_input['posz'] = (df_input['posz'] - offset_p) /  np.mean(df_input['posz'][df_input['isTouching_SMAC'] == 1][ -30:])

    df_input['CPXEts'] = df_input['CPXEts'] - df_input['CPXEts'].min()

    posx = np.round(np.mean(df_input['posx'][df_input['isArrived_Festo'] == 1] + df_input['posx_d'][
        df_input['isArrived_Festo'] == 1] / 1000))
    posy = np.round(np.mean(df_input['posy'][df_input['isArrived_Festo'] == 1] + df_input['posy_d'][
        df_input['isArrived_Festo'] == 1] / 1000))

    if posx < 50 or posy < 50:
        return None

    tuple = extract_features(df_input)
    if tuple is None:
        return None

    (stiffness, tau, F_ss, entropy, upstroke, downstroke, fi, pi, time_i, P_ss,
    force_max, time_to_max, force_overshoot, force_relaxation,
    force_rms, position_rms, damping_coefficient, elastic_coefficient, contact_area,
    force_ptp, position_ptp, load_duration,
    activity, mobility, complexity_value,
    tkeo_mean, correlation_fp, peak_ratio,sample_entropy, force_oscillation
     ) = tuple

    label = get_label(posx, posy, centers, radii, materials)

    # row = 'Test' if 110 <= posy <= 125 else 'Train'

    new_df = DataFrame({
        "posx": posx,
        "posy": posy,
        "posz": [df_input['posz'].tolist()],
        "Stiffness": stiffness,
        "Tau": tau,
        "Force Steady State": F_ss,
        "Entropy": entropy,
        "Upstroke": upstroke,
        "Downstroke": downstroke,
        "P_ss": P_ss,
        "force_max": force_max,
        "time_to_max": time_to_max,
        "force_overshoot": force_overshoot,
        "force_relaxation": force_relaxation,
        "force_oscillation": force_oscillation,
        "max_position": np.max(pi) if len(pi) > 0 else None,
        "energy_input": np.trapezoid(fi[:np.argmax(fi) + 1], pi[:np.argmax(fi) + 1]) if len(fi) > 0 and len(
            pi) > 0 else None,
        "force_rms": force_rms,
        "position_rms": position_rms,
        "damping_coefficient": damping_coefficient,
        "elastic_coeff": elastic_coefficient,
        "contact_area": contact_area,
        "force_ptp": force_ptp,
        "position_ptp": position_ptp,
        "load_duration": load_duration,
        "activity": activity,
        "mobility": mobility,
        "complexity_value": complexity_value,
        "tkeo_mean": tkeo_mean,
        "correlation_fp": correlation_fp,
        "peak_ratio": peak_ratio,
        "sample_entropy": sample_entropy,

        # Keep raw signals for later processing
        "t": [df_input['CPXEts'].tolist()],
        "Fz_s": [df_input['Fz_s'].tolist()],
        "posz_s": [df_input['posz_s'].tolist()],
        "Fi": [fi],
        "Pi": [pi],
        "Timei": [time_i],
        "label": label,
    })

    new_df = compute_hysteresis_features_for_df(new_df, force_column='Fi', pos_column='Pi')
    return new_df


def create_df(path: str = 'DamasconeA/Dataset/*.csv', symmetric=False, symm_v=False, angle=0, offset = (49, 49)) -> DataFrame | None:
    # Load JSON file

    strings = path.split('/')
    with open(strings[0]+"/"+strings[1] + "/phantom_metadata.json", "r") as file:
        json_data = json.load(file)

    # Create lists of centers + radii (for each labeled area)
    centers, radii, materials = find_inclusions(json_data, symmetric=symmetric, angle=angle, symm_v=symm_v, offset=offset)

    # Get list of all CSV files
    csv_files = glob.glob(path)

    df_list = list()

    for file in csv_files:
        df = pd.read_csv(file)
        flag = (df['isTouching_SMAC'] == 0).all()
        if flag: continue
        df_ta = organize_df(df, centers, radii, materials)
        if df_ta is not None and len(df_ta) > 0:
            df_ta.insert(0, "Source", file.split("_")[-1].split(".")[0])
            df_list.append(df_ta)
        else:
            print(f"File {file} has no valid data after processing.")

    cleaned_df_list = [df.dropna(how='all') for df in df_list]
    non_empty_df_list = [df for df in cleaned_df_list if len(df) > 0]
    combined_df = pd.concat(non_empty_df_list, ignore_index=True)
    combined_df = compute_spatial_and_surface_features(combined_df)
    return combined_df
